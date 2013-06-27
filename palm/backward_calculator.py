import numpy
from palm.linalg import asym_matrix_vector_product, matrix_vector_product
from palm.util import ALMOST_ZERO

class BackwardCalculator(object):
    """
    Computes the product of a matrix exponential, a matrix, and a vector.

    Parameters
    ----------
    expm_calculator : object
        A matrix exponential calculator object.
    """
    def __init__(self, expm_calculator):
        super(BackwardCalculator, self).__init__()
        self.expm_calculator = expm_calculator

    def compute_backward_vector(self, next_prob_vec, rate_matrix_aa,
                                rate_matrix_ab, dwell_time):
        """
        Computes the product of a matrix exponential and a vector.

        ``exp(rate_matrix_aa * dwell_time) * rate_matrix_ab * next_prob_vec``

        Parameters
        ----------
        next_prob_vec : ProbabilityVector
            The probability of ending up in each state of aggregate `b`.
        rate_matrix_aa : RateMatrix
            Take the exponential of this matrix.
        rate_matrix_ab : RateMatrix
            Represents transitions from aggregate `a` to aggregate `b`.
        dwell_time : float
            The time spent in aggregate `a`.

        Returns
        -------
        bwd_vec : ProbabilityVector
            The product of the terms, as described above.
        """
        if rate_matrix_ab:
            bwd_vec = asym_matrix_vector_product(
                        rate_matrix_ab, next_prob_vec, do_alignment=True)
        else:
            bwd_vec = next_prob_vec
        bwd_vec = self.expm_calculator.compute_matrix_expv(
                          rate_matrix_aa, dwell_time, bwd_vec)
        return bwd_vec
