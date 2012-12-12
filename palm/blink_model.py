import numpy
import scipy.linalg
from base.model import Model

class BlinkState(object):
    def __init__(self, id_str, I, A, D, B, obs_class):
        self.id = id_str
        self.I = I
        self.A = A
        self.D = D
        self.B = B
        self.obs_class = obs_class
        self.is_start_state = False
        self.is_end_state = False
    def __str__(self):
        return self.id
    def get_vec(self):
        return numpy.array([self.I, self.A, self.D, self.B])


class BlinkRoute(object):
    def __init__(self, start_state, end_state, log_rate,
                 log_combinatoric_factor):
        self.start_state = start_state
        self.end_state = end_state
        self.log_rate = log_rate
        self.log_combinatoric_factor = log_combinatoric_factor
    def __str__(self):
        return "%s --> %s, %d  %.2f" % (self.start_state, self.end_state,
                                        self.log_combinatoric_factor,
                                        self.log_rate)


class TimeVaryingBlinkRoute(object):
    def __init__(self, start_state, end_state, log_rate_fcn,
                 log_combinatoric_factor):
        self.start_state = start_state
        self.end_state = end_state
        self.log_rate_fcn = log_rate_fcn
        self.log_combinatoric_factor = log_combinatoric_factor
    def __str__(self):
        return "%s --> %s, %d  %.2f" % (self.start_state, self.end_state,
                                        self.log_combinatoric_factor,
                                        self.log_rate_fcn(0.))


class BlinkModel(object):
    def __init__(self, state_enumerator, route_mapper, parameter_set):
        self.state_enumerator = state_enumerator
        self.route_mapper = route_mapper
        self.parameter_set = parameter_set
        self.states = self.state_enumerator()

        self.dark_inds = []
        self.bright_inds = []
        for i, s in enumerate(self.states):
            if s.is_start_state:
                self.start_state = s
            if s.is_end_state:
                self.end_state = s
            if s.obs_class == 'dark':
                self.dark_inds.append(i)
            elif s.obs_class == 'bright':
                self.bright_inds.append(i)
            else:
                print "Unexpected observation class for state", s

        sorted_state_list = []
        for i in self.dark_inds:
            sorted_state_list.append(self.states[i])
        for j in self.bright_inds:
            sorted_state_list.append(self.states[j])
        sorted_dark_inds = range(len(self.dark_inds))
        sorted_bright_inds = range(len(self.dark_inds), len(sorted_state_list))
        self.states = sorted_state_list
        self.dark_inds = sorted_dark_inds
        self.bright_inds = sorted_bright_inds

        self.state_index_dict = {}
        for i, s in enumerate(self.states):
            self.state_index_dict[s.id] = i

        self.state_dict = {}
        for s in self.states:
            self.state_dict[s.id] = s

        self.route_class_dict = {"dark_dark":(self.dark_inds, self.dark_inds),
                                 "dark_bright":(self.dark_inds, self.bright_inds),
                                 "bright_dark":(self.bright_inds, self.dark_inds),
                                 "bright_bright":(self.bright_inds, self.bright_inds)}

        self.static_routes, self.time_varying_routes = self.route_mapper(self.states, self.state_index_dict)

    def __len__(self):
        return self.get_num_states()

    def __str__(self):
        full_str = ""
        full_str += "Dark States\n"
        for i in self.dark_inds:
            s = self.states[i]
            full_str += "%d  %s\n" % (i, str(s))
        full_str += "Bright States\n"
        for j in self.bright_inds:
            s = self.states[j]
            full_str += "%d  %s\n" % (j, str(s))
        full_str += "Static Routes\n"
        for r in self.static_routes:
            full_str += str(r) + '\n'
        full_str += "Time Varying Routes\n"
        for r in self.time_varying_routes:
            full_str += str(r) + '\n'
        return full_str

    def get_num_states(self):
        return len(self.states)

    def get_num_routes(self):
        return len(self.static_routes) + len(self.time_varying_routes)

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

    def get_submatrix(self, start_class, end_class):
        assert start_class in ['dark', 'bright']
        assert end_class in ['dark', 'bright']
        route_class_str = "%s_%s" % (start_class, end_class)
        route_inds = self.route_class_dict[route_class_str]
        row_inds = route_inds[0]
        col_inds = route_inds[1]
        submatrix = self.rate_matrix[row_inds[0]:row_inds[-1]+1, col_inds[0]:col_inds[-1]+1]
        return numpy.atleast_2d(submatrix)

    def build_rate_matrix(self, time=0.):
        N = len(self)
        rate_matrix = numpy.zeros([N, N])
        for r in self.static_routes:
            start_index = self.state_index_dict[r.start_state]
            end_index = self.state_index_dict[r.end_state]
            this_log_rate = r.log_combinatoric_factor + r.log_rate
            this_rate = 10**(this_log_rate)
            rate_matrix[start_index,end_index] = this_rate # Q[start,end] == start-->end

        for r in self.time_varying_routes:
            start_index = self.state_index_dict[r.start_state]
            end_index = self.state_index_dict[r.end_state]
            this_log_rate = r.log_combinatoric_factor + r.log_rate_fcn(time)
            this_rate = 10**(this_log_rate)
            rate_matrix[start_index,end_index] = this_rate # Q[start,end] == start-->end

        for i in range(N):
            rate_matrix[i,i] = -numpy.sum(rate_matrix[i,:])

        rate_matrix = numpy.asmatrix(rate_matrix)
        self.rate_matrix = rate_matrix
        return rate_matrix

    def get_init_pop(self, init_class):
        assert init_class in ['dark'], init_class
        init_pop = numpy.zeros([1,len(self.dark_inds)])
        for i in self.dark_inds:
            s = self.states[i]
            if s.I == self.N:
                all_inactive_state_index = i
                break
        init_pop[0,all_inactive_state_index] = self.N
        return numpy.asmatrix(init_pop)

    def get_class_inds(self, class_id):
        if class_id == 'dark':
            return self.dark_inds
        elif class_id == 'bright':
            return self.bright_inds
        else:
            print "Unknown class", class_id
            return None
