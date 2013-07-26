import numpy
import scipy.linalg
from pandas import DataFrame
from palm.util import DATA_TYPE

def make_rate_matrix_from_state_ids(index_id_collection, column_id_collection):
    rm = RateMatrix()
    index_id_list = index_id_collection.as_list()
    column_id_list = column_id_collection.as_list()
    rm.data_frame = DataFrame(0.0, index=index_id_list, columns=column_id_list)
    return rm

def make_rate_matrix_from_panda_data_frame(data_frame):
    rm = RateMatrix()
    rm.data_frame = data_frame
    return rm


class RateMatrix(object):
    """docstring for RateMatrix"""
    def __init__(self):
        super(RateMatrix, self).__init__()
        self.data_frame = None
    def __len__(self):
        return len(self.data_frame)
    def __str__(self):
        return str(self.data_frame)
    def __iter__(self):
        for column_id, row_series in self.data_frame.iteritems():
            yield column_id, row_series
    def get_shape(self):
        return self.data_frame.shape
    def set_rate(self, state_id1, state_id2, rate):
        self.data_frame.set_value(index=state_id1, col=state_id2, value=rate)
    def get_rate(self, state_id1, state_id2):
        return self.data_frame.get_value(index=state_id1, col=state_id2)
    def balance_transition_rates(self):
        # set diagonals to -sum of other entries in row
        diagonal_inds = numpy.diag_indices_from(self.data_frame.values)
        sum_along_row_series = self.data_frame.sum(1)
        self.data_frame.values[diagonal_inds] = -sum_along_row_series
    def as_npy_array(self):
        return self.data_frame.values
    def get_submatrix(self, index_id_collection, column_id_collection):
        sub_df = self.data_frame.reindex(
                    index=index_id_collection.as_list(),
                    columns=column_id_collection.as_list())
        return make_rate_matrix_from_panda_data_frame(sub_df)
    def get_index_id_list(self):
        return self.data_frame.index.tolist()
    def get_column_id_list(self):
        return self.data_frame.columns.tolist()
    def copy(self):
        return make_rate_matrix_from_panda_data_frame(self.data_frame.copy())
    def compute_sparsity(self):
        zero_df = self.data_frame[self.data_frame == 0.0]
        num_zero_entries = zero_df.count().sum()
        num_nonzero = numpy.count_nonzero(self.data_frame.values)
        total_entries = len(self) * len(self)
        assert total_entries == (num_nonzero + num_zero_entries)
        fraction_zero = num_zero_entries / (1. * total_entries)
        print total_entries, num_zero_entries
        return fraction_zero
    def compute_norm(self):
        return scipy.linalg.norm(self.data_frame.values, numpy.inf)
    def compute_max_element_magnitude(self):
        return numpy.max(numpy.abs(self.data_frame.values))
    def print_non_zero_entries(self):
        r,c = numpy.where(self.data_frame.values > 0.0)
        nonzero_values = numpy.ravel(self.data_frame.values[r,c])
        nonzero_values.sort()
        reversed_nonzero_values = nonzero_values[::-1]
        print reversed_nonzero_values[:min(len(self),30)]
    def is_finite(self):
        """
        Returns
        -------
        all_finite : True if no elements of vector are inf, -inf, or nan.
        """
        all_finite = numpy.all(numpy.isfinite(self.data_frame))
        return all_finite


class RateMatrixTrajectory(object):
    """docstring for RateMatrixTrajectory"""
    def __init__(self):
        super(RateMatrixTrajectory, self).__init__()
        self.matrix_list = []
    def __iter__(self):
        for m in iter(self.matrix_list):
            yield m
    def __len__(self):
        return len(self.matrix_list)
    def add_matrix(self, rate_matrix):
        self.matrix_list.append(rate_matrix)
