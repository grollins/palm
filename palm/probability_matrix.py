import numpy
from collections import defaultdict

class ProbabilityMatrix(object):
    """docstring for ProbabilityMatrix"""
    def __init__(self):
        super(ProbabilityMatrix, self).__init__()
        self.state_id_collection = None
        self.transition_probability_by_id_dict = defaultdict(dict)
    def __len__(self):
        return len(self.state_id_collection)
    def load_ids_and_numpy_array(self, state_id_collection, npy_product):
        self.state_id_collection = state_id_collection
        for r, r_id in enumerate(state_id_collection):
            for c, c_id in enumerate(state_id_collection):
                self.transition_probability_by_id_dict[r_id][c_id] = npy_product[r,c]
    def add_state_ids(self, state_id_collection):
        self.state_id_collection = state_id_collection
    def get_state_id_collection(self):
        return self.state_id_collection
    def set_transition_probability(self, state_id1, state_id2, transition_prob):
        self.transition_probability_by_id_dict[state_id1][state_id2] = transition_prob
    def balance_transition_prob(self):
        for s_id in self.state_id_collection:
            transitions = self.transition_probability_by_id_dict[s_id]
            trans_prob_sum = numpy.array(transitions.values()).sum()
            self.transition_probability_by_id_dict[s_id][s_id] = 1 - trans_prob_sum
    def as_numpy_array(self, state_id_collection):
        N = len(state_id_collection)
        my_array = numpy.zeros( [N,N] )
        for r, r_id in enumerate(state_id_collection):
            for c, c_id in enumerate(state_id_collection):
                my_array[r,c] = self.transition_probability_by_id_dict[r_id][c_id]
        return my_array
