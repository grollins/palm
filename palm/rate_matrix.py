import numpy
from palm.util import DATA_TYPE

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

    def get_numpy_submatrix(self, start_class, end_class):
        row_inds = self.class_indices_dict[start_class]
        col_inds = self.class_indices_dict[end_class]
        submatrix = self.rate_matrix[row_inds[0]:row_inds[-1]+1,
                                     col_inds[0]:col_inds[-1]+1]
        return numpy.atleast_2d(submatrix)
