import numpy
from palm.aggregated_kinetic_model import AggregatedKineticModel

class SingleDarkState(object):
    '''
    A state class for use with an aggregated kinetic model.
    The available microstates are I, A, D, and B.
    '''
    def __init__(self, id_str, I, A, D, B, observation_class):
        self.id = id_str
        self.I = I
        self.A = A
        self.D = D
        self.B = B
        self.observation_class = observation_class
        self.initial_state_flag = False

    def __str__(self):
        return "%s %s" % (self.id, self.observation_class)

    def as_array(self):
        return numpy.array([self.I, self.A, self.D, self.B])

    def get_id(self):
        return self.id

    def get_class(self):
        return self.observation_class

    def is_initial_state(self):
        return self.initial_state_flag

    def set_initial_state_flag(self):
        self.initial_state_flag = True


class DoubleDarkState(object):
    '''
    A state class for use with an aggregated kinetic model.
    The available microstates are I, A, D1, D2, and B.
    '''
    def __init__(self, id_str, I, A, D1, D2, B, observation_class):
        self.id = id_str
        self.I = I
        self.A = A
        self.D1 = D1
        self.D2 = D2
        self.B = B
        self.observation_class = observation_class
        self.initial_state_flag = False

    def __str__(self):
        return "%s %s" % (self.id, self.observation_class)

    def as_array(self):
        return numpy.array([self.I, self.A, self.D1, self.D2, self.B])

    def get_id(self):
        return self.id

    def get_class(self):
        return self.observation_class

    def is_initial_state(self):
        return self.initial_state_flag

    def set_initial_state_flag(self):
        self.initial_state_flag = True


class BlinkModel(AggregatedKineticModel):
    '''
    BlinkModel is an AggregatedKineticModel. Two observation classes
    are expected:
    1. dark (no fluorescence detected)
    2. bright (fluorescence detected)
    '''
    def __init__(self, state_enumerator, route_mapper, parameter_set):
        super(BlinkModel, self).__init__(state_enumerator, route_mapper,
                                          parameter_set)

    def get_initial_population_array(self):
        initial_population_array = numpy.zeros([1, self.get_num_states()])
        initial_population_array[0, self.initial_state_index] = 1.0
        dark_inds = self.class_indices_dict['dark']
        start_col = dark_inds[0]
        end_col = dark_inds[-1]
        return initial_population_array[:, start_col:end_col+1]
