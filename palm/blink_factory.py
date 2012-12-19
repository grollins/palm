import numpy
import blink_model
from boop.util import n_choose_k, multichoose
import aggregated_kinetic_model
import base.model_factory
from types import IntType

MAX_A = 10

class Transition(object):
    def __init__(self, dI, dA, dD, dB, label, reacting_species_dict,
                 log_rate_fcn_factory):
        self.dPop_dict = {'I':dI, 'A':dA, 'D':dD, 'B':dB}
        self.reacting_species_dict = reacting_species_dict
        self.label = label
        self.log_rate_fcn_factory = log_rate_fcn_factory

    def __str__(self):
        return "%s  %d_%d_%d_%d" % (self.label,
                                    self.dPop_dict['I'],
                                    self.dPop_dict['A'],
                                    self.dPop_dict['D'],
                                    self.dPop_dict['B'])

    def get_dPop(self, species_label):
        return self.dPop_dict[species_label]

    def is_allowed(self, start_state):
        return_value = True
        for rs in self.reacting_species_dict.iterkeys():
            num_reactants = self.reacting_species_dict[rs]
            if rs == 'I':
                species_starting_pop = start_state.I
            elif rs == 'A':
                species_starting_pop = start_state.A
            elif rs == 'D':
                species_starting_pop = start_state.D
            elif rs == 'B':
                species_starting_pop = start_state.B
            if species_starting_pop < num_reactants:
                # we need at least num_reactants for the transition
                return_value = False
                break
        return return_value

    def compute_log_combinatoric_factor(self, current_pop_dict):
        rs = self.reacting_species_dict.keys()[0]
        n = current_pop_dict[rs]
        k = abs(self.reacting_species_dict[rs])
        combinatoric_factor = n_choose_k(n,k)
        return numpy.log10(combinatoric_factor)


class BlinkModelFactory(base.model_factory.ModelFactory):
    def __init__(self):
        self.state_factory = blink_model.BlinkState
        self.route_factory = aggregated_kinetic_model.Route

    def create_model(self, parameter_set):
        self.parameter_set = parameter_set
        state_enumerator = self.state_enumerator_factory()
        route_mapper = self.route_mapper_factory()
        new_model = blink_model.BlinkModel(state_enumerator, route_mapper,
                                           self.parameter_set)
        return new_model

    def state_enumerator_factory(self):
        N = self.parameter_set.get_parameter('N')
        assert type(N) is IntType
        def enumerate_states():
            # There are 4 states (I, A, D, B)
            state_list = []
            i = 0
            for this_count_list in multichoose(4,N):
                I = this_count_list[0]
                A = this_count_list[1]
                D = this_count_list[2]
                B = this_count_list[3]
                if A > MAX_A:
                    continue
                else:
                    if A > 0:
                        obs_class = 'bright'
                    else:
                        obs_class = 'dark'
                    id_str = "%d_%d_%d_%d" % (I, A, D, B)
                    this_state = self.state_factory(id_str, I, A, D, B,
                                                    obs_class)
                    if I == N:
                        this_state.set_initial_state_flag()
                    state_list.append(this_state)
                    i += 1
            return state_list
        return enumerate_states

    def log_ka_factory(self, log_combinatoric_factor):
        log_ka = self.parameter_set.get_parameter('log_ka')
        def log_ka_fcn(t):
            return log_ka + log_combinatoric_factor
        return log_ka_fcn

    def log_kd_factory(self, log_combinatoric_factor):
        log_kd = self.parameter_set.get_parameter('log_kd')
        def log_kd_fcn(t):
            return log_kd + log_combinatoric_factor
        return log_kd_fcn

    def log_kr_factory(self, log_combinatoric_factor):
        log_kr = self.parameter_set.get_parameter('log_kr')
        def log_kr_fcn(t):
            return log_kr + log_combinatoric_factor
        return log_kr_fcn

    def log_kb_factory(self, log_combinatoric_factor):
        log_kb = self.parameter_set.get_parameter('log_kb')
        def log_kb_fcn(t):
            return log_kb + log_combinatoric_factor
        return log_kb_fcn

    def enumerate_allowed_transitions(self, s1, allowed_transitions_list):
        for t in allowed_transitions_list:
            I2 = s1.I + t.get_dPop('I')
            A2 = s1.A + t.get_dPop('A')
            D2 = s1.D + t.get_dPop('D')
            B2 = s1.B + t.get_dPop('B')
            s2_array = numpy.array([I2, A2, D2, B2])
            no_negative_pop = len(numpy.where(s2_array < 0)[0]) == 0
            if A2 <= MAX_A and no_negative_pop and t.is_allowed(s1):
                yield s2_array, t

    def route_mapper_factory(self):
        allowed_transitions_list = [ Transition(-1, 1, 0, 0, 'I->A', {'I':1},                                               self.log_ka_factory), # activate
                                     Transition(0, -1, 1, 0, 'A->D', {'A':1},                                               self.log_kd_factory),     # blink
                                     Transition(0, -1, 0, 1, 'A->B', {'A':1},                                               self.log_kb_factory),    # photobleach
                                     Transition(0, 1, -1, 0, 'D->A', {'D':1},                                               self.log_kr_factory)]    # unblink,reactivate

        def map_routes(state_list, state_index_dict):
            route_list = []
            for s1 in state_list:
                for s2_array, t in self.enumerate_allowed_transitions(s1, allowed_transitions_list):
                    current_population_dict = {'I':s1.I, 'A':s1.A, 'D':s1.D, 'B':s1.B}
                    log_combinatoric_factor = t.compute_log_combinatoric_factor(current_population_dict)
                    start_id = s1.id
                    end_id = "%d_%d_%d_%d" % (s2_array[0], s2_array[1],
                                              s2_array[2], s2_array[3])
                    log_rate_fcn = t.log_rate_fcn_factory(log_combinatoric_factor)
                    new_route = self.route_factory(start_id, end_id, log_rate_fcn)
                    route_list.append(new_route)
            return route_list
        return map_routes

