import numpy

class ProbabilityVector(object):
    """docstring for ProbabilityVector"""
    def __init__(self):
        super(ProbabilityVector, self).__init__()
        self.state_id_collection = None
        self.state_probability_by_id_dict = {}
    def __len__(self):
        return len(self.state_id_collection)
    def __str__(self):
        my_str = ""
        for s_id in self.state_id_collection:
            my_str += "%s %.2e\n" % (s_id, self.get_state_probability(s_id))
        return my_str
    def add_state_ids(self, state_id_collection):
        self.state_id_collection = state_id_collection
    def get_state_id_collection(self):
        return self.state_id_collection
    def set_state_probability(self, state_id, probability):
        assert state_id in self.state_id_collection
        self.state_probability_by_id_dict[state_id] = probability
    def get_state_probability(self, state_id):
        return self.state_probability_by_id_dict[state_id]
    def set_uniform_state_probability(self):
        num_states = len(self)
        uniform_pop = 1./num_states
        for s_id in self.state_id_collection:
            self.state_probability_by_id_dict[s_id] = uniform_pop
    def sum_vector(self):
        vec_sum = numpy.array(self.state_probability_by_id_dict.values()).sum()
        return vec_sum
    def scale_vector(self, scale_factor):
        for k in self.state_probability_by_id_dict.iterkeys():
            self.state_probability_by_id_dict[k] *= scale_factor
    def as_numpy_row_array(self, state_id_collection):
        pop_array = numpy.zeros( [1, len(state_id_collection)] )
        pop_list = []
        for s_id in state_id_collection:
            pop_list.append(self.get_state_probability(s_id))
        pop_array[0,:] = numpy.array(pop_list)
        return pop_array
    def as_numpy_column_array(self, state_id_collection):
        return self.as_numpy_row_array(state_id_collection).T
