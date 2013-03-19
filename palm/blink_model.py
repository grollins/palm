import numpy
from palm.aggregated_kinetic_model import AggregatedKineticModel
from palm.probability_vector import make_prob_vec_from_state_ids

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

class StateIDCollection(object):
    """docstring for StateIDCollection"""
    def __init__(self):
        super(StateIDCollection, self).__init__()
        self.state_id_list = []
    def __str__(self):
        return str(self.state_id_list)
    def __iter__(self):
        for s in self.state_id_list:
            yield s
    def __contains__(self, state_id):
        return (state_id in self.state_id_list)
    def __len__(self):
        return len(self.state_id_list)
    def add_id(self, state_id):
        self.state_id_list.append(state_id)
    def as_list(self):
        return self.state_id_list

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
        self.dark_state_id_collection = None
        self.bright_state_id_collection = None
        self._fill_state_id_collections()

    def _fill_state_id_collections(self):
        self.dark_state_id_collection = StateIDCollection()
        for dark_state in self.iter_dark_states():
            self.dark_state_id_collection.add_id( dark_state.get_id() )
        self.bright_state_id_collection = StateIDCollection()
        for bright_state in self.iter_bright_states():
            self.bright_state_id_collection.add_id( bright_state.get_id() )

    def iter_dark_states(self):
        dark_inds = self.class_indices_dict['dark']
        for ind in dark_inds:
            s = self.states[ind]
            yield s

    def iter_bright_states(self):
        bright_inds = self.class_indices_dict['bright']
        for ind in bright_inds:
            s = self.states[ind]
            yield s

    def get_initial_probability_vector(self):
        initial_prob_vec = make_prob_vec_from_state_ids(
                            self.dark_state_id_collection)
        initial_prob_vec.set_state_probability(self.all_inactive_state_id, 1.0)
        return initial_prob_vec
