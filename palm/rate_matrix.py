import numpy
from pandas import DataFrame
from palm.util import DATA_TYPE

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


class RateMatrix(object):
    """docstring for RateMatrix"""
    def __init__(self):
        super(RateMatrix, self).__init__()
        self.date_frame = None
    def __len__(self):
        return len(self.data_frame)
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
    def get_submatrix(self, index_id_collection, column_id_collection):
        sub_df = self.data_frame.reindex(
                    index=index_id_collection.as_list(),
                    columns=column_id_collection.as_list())
        return make_rate_matrix_from_panda_data_frame(sub_df)


class RateMatrixFactory(object):
    """
    This factory class creates a rate matrix for an
    aggregated kinetic model. States and routes must
    be defined in order to create the matrix.
    """
    def __init__(self, rate_matrix_class):
        super(RateMatrixFactory, self).__init__()
        self.rate_matrix_class = rate_matrix_class

    def create_rate_matrix(self, model_size, model_routes, state_index_dict,
                            class_indices_dict, time):
        rate_matrix = self.rate_matrix_class( model_size, class_indices_dict )
        for r in model_routes:
            start_index = state_index_dict[r.start_state]
            end_index = state_index_dict[r.end_state]
            this_log_rate = r.compute_log_rate(t=time)
            this_rate = 10**(this_log_rate)
            assert this_rate >= 0.0, "%d %d %s" % (start_index, end_index,
                                                   str(this_rate))
            rate_matrix.set_rate(start_index, end_index, this_rate)
        rate_matrix.finalize_matrix()
        return rate_matrix

    def create_submatrix(self, rate_matrix, start_class, end_class):
        row_inds = rate_matrix.class_indices_dict[start_class]
        col_inds = rate_matrix.class_indices_dict[end_class]
        numpy_submatrix = rate_matrix.rate_matrix[row_inds[0]:row_inds[-1]+1,
                                                  col_inds[0]:col_inds[-1]+1]
        numpy_submatrix = numpy.atleast_2d(numpy_submatrix)
        submatrix = self.rate_matrix_class(
                        model_size=1, class_indices_dict=None )
        submatrix.rate_matrix = numpy_submatrix
        return submatrix


class AggregatedRateMatrix(object):
    """
    A rate matrix for a model with discrete observation classes.
    The states are grouped by observation class to give the matrix
    the following sub-matrix structure:
           Q_dd | Q_db
     Q =   -----------
           Q_bd | Q_bb
    """
    def __init__(self, model_size, class_indices_dict):
        super(AggregatedRateMatrix, self).__init__()
        self.rate_matrix = numpy.zeros((model_size,model_size), dtype=DATA_TYPE)
        self.class_indices_dict = class_indices_dict
        self.is_finalized = False

    def set_rate(self, start, end, rate):
        if self.is_finalized:
            assert False, "Rate matrix is already finalized."
        else:
            self.rate_matrix[start, end] = rate

    def finalize_matrix(self):
        if self.is_finalized:
            assert False, "Rate matrix is already finalized."
        else:
            for i in xrange(self.rate_matrix.shape[0]):
                self.rate_matrix[i,i] = -numpy.sum(self.rate_matrix[i,:])
            self.is_finalized = True

    def get_size(self, axis=0):
        return self.rate_matrix.shape[axis]

    def as_numpy_array(self):
        return self.rate_matrix

    def get_diagonal_vector(self):
        return self.rate_matrix.diagonal()
