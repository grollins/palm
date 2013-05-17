import numpy
from palm.linalg import asym_matrix_vector_product, matrix_vector_product,\
                        QitMatrixExponential, ScipyMatrixExponential2,\
                        StubExponential
from palm.util import ALMOST_ZERO

class BackwardCalculator(object):
    """docstring for BackwardCalculator"""
    def __init__(self, expm_calculator):
        super(BackwardCalculator, self).__init__()
        self.expm_calculator = expm_calculator
        self.backup_calculator = ScipyMatrixExponential2()

    def compute_backward_vector(self, next_prob_vec, rate_matrix_aa,
                                rate_matrix_ab, dwell_time):
        if rate_matrix_ab:
            bwd_vec = asym_matrix_vector_product(
                        rate_matrix_ab, next_prob_vec, do_alignment=True)
        else:
            bwd_vec = next_prob_vec

        #bwd_vec.fill_zeros(ALMOST_ZERO)
        #bwd_vec.fill_na(ALMOST_ZERO)
        #bwd_vec.fill_negative(ALMOST_ZERO)
        #bwd_vec.fill_positive(1.0)

        if type(self.expm_calculator) == QitMatrixExponential:
            try:
                bwd_vec = self.expm_calculator.compute_matrix_expv(
                            rate_matrix_aa, dwell_time, bwd_vec)
            except RuntimeError:
                #print bwd_vec
                raise
                #expQt = self.backup_calculator.compute_matrix_exp(
                        #rate_matrix_aa, dwell_time)
                #bwd_vec = matrix_vector_product(
                        #expQt, bwd_vec, do_alignment=True)
        elif type(self.expm_calculator) == StubExponential:
            bwd_vec = self.expm_calculator.compute_matrix_expv(
                        rate_matrix_aa, dwell_time, bwd_vec)
        else:
            expQt = self.expm_calculator.compute_matrix_exp(
                        rate_matrix_aa, dwell_time)
            bwd_vec = matrix_vector_product(
                        expQt, bwd_vec, do_alignment=True)
        return bwd_vec
