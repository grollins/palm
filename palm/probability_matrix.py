import numpy
from pandas import DataFrame
from collections import defaultdict

def make_prob_matrix_from_state_ids(state_id_collection):
    pm = ProbabilityMatrix()
    state_id_list = state_id_collection.as_list()
    pm.data_frame = DataFrame(0.0, index=state_id_list,
                              columns=state_id_list)
    return pm

def make_prob_matrix_from_panda_data_frame(date_frame):
    pm = ProbabilityMatrix()
    pm.data_frame = data_frame
    return pm

class ProbabilityMatrix(object):
    """docstring for ProbabilityMatrix"""
    def __init__(self):
        super(ProbabilityMatrix, self).__init__()
        self.data_frame = None
    def set_transition_probability(self, state_id1, state_id2,
                                   transition_prob):
        self.data_frame.set_value(index=state_id1,col=state_id2,
                                  value=transition_prob)
    def balance_transition_prob(self):
        # set diagonals to 1 - sum of other entries in row
        diagonal_inds = numpy.diag_indices_from(self.data_frame.values)
        sum_along_row_series = self.data_frame.sum(1)
        self.data_frame.values[diagonal_inds] = 1 - sum_along_row_series
