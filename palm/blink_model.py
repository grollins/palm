import numpy
from palm.aggregated_kinetic_model import AggregatedKineticModel
from palm.probability_vector import make_prob_vec_from_state_ids
from palm.state_collection import StateIDCollection

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
    def as_dict(self):
        return {'observation_class':self.get_class(),
                'I':self.I, 'A':self.A, 'D':self.D, 'B':self.B}


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
    def as_dict(self):
        return {'observation_class':self.get_class(),
                'I':self.I, 'A':self.A, 'D1':self.D1, 'D2':self.D2, 'B':self.B}


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
        self.all_inactive_state_id = self.initial_state_id

    def get_initial_probability_vector(self):
        dark_state_id_collection = self.state_ids_by_class_dict['dark']
        initial_prob_vec = make_prob_vec_from_state_ids(dark_state_id_collection)
        initial_prob_vec.set_state_probability(self.all_inactive_state_id, 1.0)
        return initial_prob_vec
