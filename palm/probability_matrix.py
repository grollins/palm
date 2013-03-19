import numpy
from pandas import DataFrame
from collections import defaultdict

def make_prob_matrix_from_state_ids(index_id_collection,
                                    column_id_collection=None):
    pm = ProbabilityMatrix()
    index_id_list = index_id_collection.as_list()
    if column_id_collection:
        column_id_list = column_id_collection.as_list()
    else:
        column_id_list = index_id_list
    pm.data_frame = DataFrame(0.0, index=index_id_list,
                              columns=column_id_list)
    return pm

def make_prob_matrix_from_panda_data_frame(data_frame):
    pm = ProbabilityMatrix()
    pm.data_frame = data_frame
    return pm

def make_rate_matrix_from_state_ids(index_id_collection,
                                    column_id_collection=None):
    pm = RateMatrix()
    index_id_list = index_id_collection.as_list()
    if column_id_collection:
        column_id_list = column_id_collection.as_list()
    else:
        column_id_list = index_id_list
    pm.data_frame = DataFrame(0.0, index=index_id_list,
                              columns=column_id_list)
    return pm

def make_rate_matrix_from_panda_data_frame(data_frame):
    pm = RateMatrix()
    pm.data_frame = data_frame
    return pm

class ProbabilityMatrix(object):
    """docstring for ProbabilityMatrix"""
    def __init__(self):
        super(ProbabilityMatrix, self).__init__()
        self.data_frame = None
    def __str__(self):
        return str(self.data_frame)
    def set_probability(self, state_id1, state_id2, prob):
        self.data_frame.set_value(index=state_id1,col=state_id2, value=prob)
    def get_probability(self, state_id1, state_id2):
        return self.data_frame.get_value(index=state_id1, col=state_id2)
    def balance_transition_prob(self):
        # set diagonals to 1 - sum of other entries in row
        diagonal_inds = numpy.diag_indices_from(self.data_frame.values)
        sum_along_row_series = self.data_frame.sum(1)
        self.data_frame.values[diagonal_inds] = 1 - sum_along_row_series
    def as_npy_array(self):
        return self.data_frame.values


class RateMatrix(object):
    """docstring for RateMatrix"""
    def __init__(self):
        super(RateMatrix, self).__init__()
        self.date_frame = None
    def __str__(self):
        return str(self.data_frame)
    def set_rate(self, state_id1, state_id2, rate):
        self.data_frame.set_value(index=state_id1,col=state_id2, value=rate)
    def get_rate(self, state_id1, state_id2):
        return self.data_frame.get_value(index=state_id1, col=state_id2)
    def balance_transition_rates(self):
        # set diagonals to -sum of other entries in row
        diagonal_inds = numpy.diag_indices_from(self.data_frame.values)
        sum_along_row_series = self.data_frame.sum(1)
        self.data_frame.values[diagonal_inds] = -sum_along_row_series
    def as_npy_array(self):
        return self.data_frame.values
