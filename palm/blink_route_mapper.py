import numpy
from palm.util import n_choose_k
from palm.route_collection import RouteCollectionFactory

class Route(object):
    '''
    A generic route class for aggregated kinetic models.
    '''
    def __init__(self, id_str, start_state_id, end_state_id, rate_id,
                 multiplicity):
        self.id = id_str
        self.start_state_id = start_state_id
        self.end_state_id = end_state_id
        self.rate_id = rate_id
        self.multiplicity = multiplicity

    def __str__(self):
        my_str = "%s %s %s %s %d" % (
                    self.id, self.start_state_id, self.end_state_id,
                    self.rate_id, self.multiplicity)
        return my_str

    def get_id(self):
        return self.id
    def get_start_state(self):
        return self.start_state_id
    def get_end_state(self):
        return self.end_state_id
    def get_multiplicity(self):
        return self.multiplicity
    def as_dict(self):
        return {'start_state':self.start_state_id,
                'end_state':self.end_state_id,
                'rate_id':self.rate_id,
                'multiplicity':self.multiplicity}


class SingleDarkRouteMapperFactory(object):
    """
    This factory class creates a route mapper for
    a blink model with one dark state.
    """
    def __init__(self, parameter_set, route_factory=Route, max_A=5,
                 fermi_activation=False):
        super(SingleDarkRouteMapperFactory, self).__init__()
        self.parameter_set = parameter_set
        self.route_factory = route_factory
        self.max_A = max_A
        self.transition_factory = SingleDarkTransition
        self.fermi_activation = fermi_activation

    def create_route_mapper(self):
        if self.fermi_activation:
            activation = self.transition_factory(
                            -1, 1, 0, 0, {'I':1}, 'fermi_ka')
        else:
            activation = self.transition_factory(
                            -1, 1, 0, 0, {'I':1}, 'ka')
        blinking = self.transition_factory(
                            0, -1, 1, 0, {'A':1}, 'kd')
        recovery = self.transition_factory(
                            0, 1, -1, 0, {'D':1}, 'kr')
        bleaching = self.transition_factory(
                            0, -1, 0, 1, {'A':1}, 'kb')
        allowed_transitions_list = [activation, blinking, recovery, bleaching]

        def map_routes(state_collection):
            rc_factory = RouteCollectionFactory()
            for start_id, start_state in state_collection.iter_states():
                route_iterator = self.enumerate_allowed_transitions(
                                    start_state, allowed_transitions_list)
                for end_id, transition in route_iterator:
                    rate_id = transition.rate_id
                    multiplicity = transition.compute_multiplicity(start_state)
                    route_id = "%s__%s" % (start_id, end_id)
                    new_route = self.route_factory(route_id, start_id, end_id,
                                                   rate_id, multiplicity)
                    rc_factory.add_route(new_route)
            route_collection = rc_factory.make_route_collection()
            return route_collection
        return map_routes

    def enumerate_allowed_transitions(self, start_state,
                                      allowed_transitions_list):
        for transition in allowed_transitions_list:
            I2 = start_state['I'] + transition.get_dPop('I')
            A2 = start_state['A'] + transition.get_dPop('A')
            D2 = start_state['D'] + transition.get_dPop('D')
            B2 = start_state['B'] + transition.get_dPop('B')
            end_state_array = numpy.array([I2, A2, D2, B2])
            no_negative_pop = len(numpy.where(end_state_array < 0)[0]) == 0
            if A2 <= self.max_A and no_negative_pop and\
              transition.is_allowed(start_state):
                end_id = "%d_%d_%d_%d" % (I2, A2, D2, B2)
                yield end_id, transition


class SingleDarkTransition(object):
    """
    A helper class for SingleDarkRouteMapperFactory.
    """
    def __init__(self, dI, dA, dD, dB, reacting_species_dict, rate_id):
        self.dPop_dict = {'I':dI, 'A':dA, 'D':dD, 'B':dB}
        self.reacting_species_dict = reacting_species_dict
        self.rate_id = rate_id

    def __str__(self):
        return "%s  %d_%d_%d_%d" % (self.label,
                                    self.dPop_dict['I'],
                                    self.dPop_dict['A'],
                                    self.dPop_dict['D'],
                                    self.dPop_dict['B'])

    def get_dPop(self, species_label):
        return self.dPop_dict[species_label]

    def is_allowed(self, state):
        return_value = True
        for rs in self.reacting_species_dict.iterkeys():
            num_reactants = self.reacting_species_dict[rs]
            if rs == 'I':
                species_starting_pop = state['I']
            elif rs == 'A':
                species_starting_pop = state['A']
            elif rs == 'D':
                species_starting_pop = state['D']
            elif rs == 'B':
                species_starting_pop = state['B']
            if species_starting_pop < num_reactants:
                # we need at least num_reactants for the transition
                return_value = False
                break
        return return_value

    def compute_multiplicity(self, start_state):
        return 10**self.compute_log_combinatoric_factor(start_state)

    def compute_log_combinatoric_factor(self, start_state):
        # reacting_species_id = I, A, D, or B
        reacting_species_id = self.reacting_species_dict.keys()[0]
        n = start_state[reacting_species_id]
        k = abs(self.reacting_species_dict[reacting_species_id])
        combinatoric_factor = n_choose_k(n,k)
        return numpy.log10(combinatoric_factor)


class DoubleDarkRouteMapperFactory(object):
    """
    This factory class creates a route mapper for
    a blink model with two dark states.
    """
    def __init__(self, parameter_set, route_factory=Route, max_A=5,
                 fermi_activation=False):
        super(DoubleDarkRouteMapperFactory, self).__init__()
        self.parameter_set = parameter_set
        self.route_factory = route_factory
        self.max_A = max_A
        self.transition_factory = DoubleDarkTransition
        self.fermi_activation = fermi_activation

    def create_route_mapper(self):
        if self.fermi_activation:
            activation = self.transition_factory(
                            -1, 1, 0, 0, 0, {'I':1}, 'fermi_ka')
        else:
            activation = self.transition_factory(
                            -1, 1, 0, 0, 0, {'I':1}, 'ka')
        blinking1 = self.transition_factory(
                            0, -1, 1, 0, 0, {'A':1}, 'kd1')
        recovery1 = self.transition_factory(
                            0, 1, -1, 0, 0, {'D1':1}, 'kr1')
        blinking2 = self.transition_factory(
                            0, -1, 0, 1, 0, {'A':1}, 'kd2')
        recovery2 = self.transition_factory(
                            0, 1, 0, -1, 0, {'D2':1}, 'kr2')
        bleaching = self.transition_factory(
                            0, -1, 0, 0, 1, {'A':1}, 'kb')
        allowed_transitions_list = [activation, blinking1, recovery1,
                                    blinking2, recovery2, bleaching]

        def map_routes(state_collection):
            rc_factory = RouteCollectionFactory()
            for start_id, start_state in state_collection.iter_states():
                route_iterator = self.enumerate_allowed_transitions(
                                    start_state, allowed_transitions_list)
                for end_id, transition in route_iterator:
                    rate_id = transition.rate_id
                    multiplicity = transition.compute_multiplicity(start_state)
                    route_id = "%s__%s" % (start_id, end_id)
                    new_route = self.route_factory(route_id, start_id, end_id,
                                                   rate_id, multiplicity)
                    rc_factory.add_route(new_route)
            route_collection = rc_factory.make_route_collection()
            return route_collection
        return map_routes

    def enumerate_allowed_transitions(self, start_state,
                                      allowed_transitions_list):
        for transition in allowed_transitions_list:
            end_I =  start_state['I'] + transition.get_dPop('I')
            end_A =  start_state['A'] + transition.get_dPop('A')
            end_D1 = start_state['D1'] + transition.get_dPop('D1')
            end_D2 = start_state['D2'] + transition.get_dPop('D2')
            end_B =  start_state['B'] + transition.get_dPop('B')
            end_state_array = numpy.array([end_I, end_A, end_D1, end_D2, end_B])
            no_negative_pop = len(numpy.where(end_state_array < 0)[0]) == 0
            if end_A <= self.max_A and no_negative_pop and\
              transition.is_allowed(start_state):
                end_id = "%d_%d_%d_%d_%d" % (end_I, end_A, end_D1, end_D2, end_B)
                yield end_id, transition


class DoubleDarkTransition(object):
    """
    A helper class for DoubleDarkRouteMapperFactory.
    """
    def __init__(self, dI, dA, dD1, dD2, dB, reacting_species_dict, rate_id):
        self.dPop_dict = {'I':dI, 'A':dA, 'D1':dD1, 'D2':dD2, 'B':dB}
        self.reacting_species_dict = reacting_species_dict
        self.rate_id = rate_id

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
                species_starting_pop = start_state['I']
            elif rs == 'A':
                species_starting_pop = start_state['A']
            elif rs == 'D1':
                species_starting_pop = start_state['D1']
            elif rs == 'D2':
                species_starting_pop = start_state['D2']
            elif rs == 'B':
                species_starting_pop = start_state['B']
            if species_starting_pop < num_reactants:
                # we need at least num_reactants for the transition
                return_value = False
                break
        return return_value

    def compute_multiplicity(self, start_state):
        return 10**self.compute_log_combinatoric_factor(start_state)

    def compute_log_combinatoric_factor(self, start_state):
        # reacting_species_id = I, A, D1, D2, or B
        reacting_species_id = self.reacting_species_dict.keys()[0]
        n = start_state[reacting_species_id]
        k = abs(self.reacting_species_dict[reacting_species_id])
        combinatoric_factor = n_choose_k(n,k)
        return numpy.log10(combinatoric_factor)
