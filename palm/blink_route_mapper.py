import numpy
import util

class SingleDarkRouteMapperFactory(object):
    """
    This factory class creates a route mapper for
    a blink model with one dark state.
    """
    def __init__(self, parameter_set, route_factory, max_A,
                 fermi_activation=False):
        super(SingleDarkRouteMapperFactory, self).__init__()
        self.parameter_set = parameter_set
        self.route_factory = route_factory
        self.max_A = max_A
        self.transition_factory = SingleDarkTransition
        self.fermi_activation = fermi_activation

    def create_route_mapper(self):
        if self.fermi_activation:
            activation = self.transition_factory(-1, 1, 0, 0, 'I->A', {'I':1},
                                                 self.fermi_log_ka_factory)
        else:
            activation = self.transition_factory(-1, 1, 0, 0, 'I->A', {'I':1},
                                                 self.log_ka_factory)            
        blinking = self.transition_factory(0, -1, 1, 0, 'A->D', {'A':1},
                                           self.log_kd_factory)
        recovery = self.transition_factory(0, 1, -1, 0, 'D->A', {'D':1},
                                           self.log_kr_factory)
        bleaching = self.transition_factory(0, -1, 0, 1, 'A->B', {'A':1},
                                            self.log_kb_factory)
        allowed_transitions_list = [activation, blinking, recovery, bleaching]

        def map_routes(state_list, state_index_dict):
            route_list = []
            for s1 in state_list:
                route_iterator = self.enumerate_allowed_transitions(s1, allowed_transitions_list)
                for s2_array, transition in route_iterator:
                    current_population_dict = {'I':s1.I, 'A':s1.A, 'D':s1.D, 'B':s1.B}
                    start_id = s1.id
                    end_id = "%d_%d_%d_%d" % (s2_array[0], s2_array[1],
                                              s2_array[2], s2_array[3])
                    log_rate_fcn = transition.get_log_rate_fcn(current_population_dict)
                    new_route = self.route_factory(start_id, end_id,
                                                   log_rate_fcn,
                                                   transition.label)
                    route_list.append(new_route)
            return route_list
        return map_routes

    def log_ka_factory(self, log_combinatoric_factor):
        log_ka = self.parameter_set.get_parameter('log_ka')
        def log_ka_fcn(t):
            return log_ka + log_combinatoric_factor
        return log_ka_fcn

    def fermi_log_ka_factory(self, log_combinatoric_factor):
        T = self.parameter_set.get_parameter('fermi_T')
        tf = self.parameter_set.get_parameter('fermi_tf')
        def fermi_fcn(t):
            numerator = numpy.exp(-(t - tf) / T)
            denominator = ((1 + numerator) * numpy.log(1 + numerator)) * T
            # if denominator < 1e-10:
            #     ka = 0.2
            # else:
            ka = numerator/denominator
            log_ka = numpy.log10(ka)
            return log_ka + log_combinatoric_factor
        return fermi_fcn

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
        for transition in allowed_transitions_list:
            I2 = s1.I + transition.get_dPop('I')
            A2 = s1.A + transition.get_dPop('A')
            D2 = s1.D + transition.get_dPop('D')
            B2 = s1.B + transition.get_dPop('B')
            s2_array = numpy.array([I2, A2, D2, B2])
            no_negative_pop = len(numpy.where(s2_array < 0)[0]) == 0
            if A2 <= self.max_A and no_negative_pop and transition.is_allowed(s1):
                yield s2_array, transition


class SingleDarkTransition(object):
    """
    A helper class for SingleDarkRouteMapperFactory.
    """
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
        combinatoric_factor = util.n_choose_k(n,k)
        return numpy.log10(combinatoric_factor)

    def get_log_rate_fcn(self, current_pop_dict):
        log_comb_factor = self.compute_log_combinatoric_factor(current_pop_dict)
        log_rate_fcn = self.log_rate_fcn_factory(log_comb_factor)
        return log_rate_fcn


class DoubleDarkRouteMapperFactory(object):
    """
    This factory class creates a route mapper for
    a blink model with two dark states.
    """
    def __init__(self, parameter_set, route_factory, max_A,
                 fermi_activation=False):
        super(DoubleDarkRouteMapperFactory, self).__init__()
        self.parameter_set = parameter_set
        self.route_factory = route_factory
        self.max_A = max_A
        self.transition_factory = DoubleDarkTransition
        self.fermi_activation = fermi_activation

    def create_route_mapper(self):
        if self.fermi_activation:
            activation = self.transition_factory(-1, 1, 0, 0, 0, 'I->A', {'I':1},
                                                 self.fermi_log_ka_factory)
        else:
            activation = self.transition_factory(-1, 1, 0, 0, 0, 'I->A', {'I':1},
                                                 self.log_ka_factory)            
        blinking1 = self.transition_factory(0, -1, 1, 0, 0, 'A->D1', {'A':1},
                                           self.log_kd1_factory)
        recovery1 = self.transition_factory(0, 1, -1, 0, 0, 'D1->A', {'D1':1},
                                           self.log_kr1_factory)
        blinking2 = self.transition_factory(0, -1, 0, 1, 0, 'A->D2', {'A':1},
                                           self.log_kd2_factory)
        recovery2 = self.transition_factory(0, 1, 0, -1, 0, 'D2->A', {'D2':1},
                                           self.log_kr2_factory)
        bleaching = self.transition_factory(0, -1, 0, 0, 1, 'A->B', {'A':1},
                                            self.log_kb_factory)
        allowed_transitions_list = [activation, blinking1, recovery1,
                                    blinking2, recovery2, bleaching]

        def map_routes(state_list, state_index_dict):
            route_list = []
            for s1 in state_list:
                route_iterator = self.enumerate_allowed_transitions(s1, allowed_transitions_list)
                for s2_array, transition in route_iterator:
                    current_population_dict = {'I':s1.I, 'A':s1.A, 'D1':s1.D1,
                                               'D2':s1.D2, 'B':s1.B}
                    start_id = s1.id
                    end_id = "%d_%d_%d_%d_%d" % (s2_array[0], s2_array[1],
                                                 s2_array[2], s2_array[3],
                                                 s2_array[4])
                    log_rate_fcn = transition.get_log_rate_fcn(current_population_dict)
                    new_route = self.route_factory(start_id, end_id,
                                                   log_rate_fcn, transition.label)
                    route_list.append(new_route)
            return route_list
        return map_routes

    def log_ka_factory(self, log_combinatoric_factor):
        log_ka = self.parameter_set.get_parameter('log_ka')
        def log_ka_fcn(t):
            return log_ka + log_combinatoric_factor
        return log_ka_fcn

    def fermi_log_ka_factory(self, log_combinatoric_factor):
        T = self.parameter_set.get_parameter('fermi_T')
        tf = self.parameter_set.get_parameter('fermi_tf')
        def fermi_fcn(t):
            numerator = numpy.exp(-(t - tf) / T)
            denominator = ((1 + numerator) * numpy.log(1 + numerator)) * T
            ka = numerator/denominator
            log_ka = numpy.log10(ka)
            return log_ka + log_combinatoric_factor
        return fermi_fcn

    def log_kd1_factory(self, log_combinatoric_factor):
        log_kd1 = self.parameter_set.get_parameter('log_kd1')
        def log_kd1_fcn(t):
            return log_kd1 + log_combinatoric_factor
        return log_kd1_fcn

    def log_kr1_factory(self, log_combinatoric_factor):
        log_kr1 = self.parameter_set.get_parameter('log_kr1')
        def log_kr1_fcn(t):
            return log_kr1 + log_combinatoric_factor
        return log_kr1_fcn

    def log_kd2_factory(self, log_combinatoric_factor):
        log_kd2 = self.parameter_set.get_parameter('log_kd2')
        def log_kd2_fcn(t):
            return log_kd2 + log_combinatoric_factor
        return log_kd2_fcn

    def log_kr2_factory(self, log_combinatoric_factor):
        log_kr_diff = self.parameter_set.get_parameter('log_kr_diff')
        log_kr1 = self.parameter_set.get_parameter('log_kr1')
        log_kr2 = log_kr1 + log_kr_diff
        def log_kr2_fcn(t):
            return log_kr2 + log_combinatoric_factor
        return log_kr2_fcn

    def log_kb_factory(self, log_combinatoric_factor):
        log_kb = self.parameter_set.get_parameter('log_kb')
        def log_kb_fcn(t):
            return log_kb + log_combinatoric_factor
        return log_kb_fcn

    def enumerate_allowed_transitions(self, start_state,
                                      allowed_transitions_list):
        for transition in allowed_transitions_list:
            end_I = start_state.I + transition.get_dPop('I')
            end_A = start_state.A + transition.get_dPop('A')
            end_D1 = start_state.D1 + transition.get_dPop('D1')
            end_D2 = start_state.D2 + transition.get_dPop('D2')
            end_B = start_state.B + transition.get_dPop('B')
            end_state_array = numpy.array([end_I, end_A, end_D1, end_D2, end_B])
            no_negative_pop = len(numpy.where(end_state_array < 0)[0]) == 0
            if end_A <= self.max_A and no_negative_pop and transition.is_allowed(start_state):
                yield end_state_array, transition


class DoubleDarkTransition(object):
    """
    A helper class for DoubleDarkRouteMapperFactory.
    """
    def __init__(self, dI, dA, dD1, dD2, dB, label, reacting_species_dict,
                 log_rate_fcn_factory):
        self.dPop_dict = {'I':dI, 'A':dA, 'D1':dD1, 'D2':dD2, 'B':dB}
        self.reacting_species_dict = reacting_species_dict
        self.label = label
        self.log_rate_fcn_factory = log_rate_fcn_factory

    def __str__(self):
        return "%s  %d_%d_%d_%d_%d" % (self.label,
                                       self.dPop_dict['I'],
                                       self.dPop_dict['A'],
                                       self.dPop_dict['D1'],
                                       self.dPop_dict['D2'],
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
            elif rs == 'D1':
                species_starting_pop = start_state.D1
            elif rs == 'D2':
                species_starting_pop = start_state.D2
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
        combinatoric_factor = util.n_choose_k(n,k)
        return numpy.log10(combinatoric_factor)

    def get_log_rate_fcn(self, current_pop_dict):
        log_comb_factor = self.compute_log_combinatoric_factor(current_pop_dict)
        log_rate_fcn = self.log_rate_fcn_factory(log_comb_factor)
        return log_rate_fcn
