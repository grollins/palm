import numpy
from palm.linalg import asym_vector_matrix_product, vector_matrix_product

class ForwardCalculator(object):
    """docstring for ForwardCalculator"""
    def __init__(self, expm_calculator):
        super(ForwardCalculator, self).__init__()
        self.expm_calculator = expm_calculator

    def compute_forward_vector(self, init_prob, rate_matrix_aa,
                               rate_matrix_ab, dwell_time):
        expQt = self.expm_calculator.compute_matrix_exp(
                    rate_matrix_aa, dwell_time)
        fwd_vec = vector_matrix_product(init_prob, expQt, do_alignment=True)
        if rate_matrix_ab:
            fwd_vec = asym_vector_matrix_product(
                        fwd_vec, rate_matrix_ab, do_alignment=True)
        else:
            pass
        return fwd_vec

    def compute_forward_state(self, init_prob, rate_matrix_aa,
                               rate_matrix_ab, dwell_time):
        fwd_vec = self.compute_forward_vector(
                    init_prob, rate_matrix_aa, rate_matrix_ab, dwell_time)
        ml_state = fwd_vec.get_ml_state()
        return ml_state
