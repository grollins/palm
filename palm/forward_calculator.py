from .linalg import asym_vector_matrix_product, vector_matrix_product, \
                    compute_inverse


class ForwardCalculator(object):
    """
    Computes the product of a vector, a matrix exponential, and a matrix.

    Parameters
    ----------
    expm_calculator : object
        A matrix exponential calculator object.
    dead_time : float
        The temporal resolution of the data. Necessary for correcting rate
        matrices for missed transitions. Units: seconds.
    """
    def __init__(self, expm_calculator, dead_time=0.05):
        super(ForwardCalculator, self).__init__()
        self.expm_calculator = expm_calculator
        self.dead_time = dead_time

    def compute_forward_vector(self, init_prob, rate_matrix_aa,
                               rate_matrix_ab, dwell_time):
        """
        Computes the product of a matrix exponential and a vector.

        ``init_prob * exp(rate_matrix_aa * dwell_time) * rate_matrix_ab``

        Parameters
        ----------
        init_prob : ProbabilityVector
            The probability of starting in each state of aggregate `a`.
        rate_matrix_aa : RateMatrix
            Take the exponential of this matrix.
        rate_matrix_ab : RateMatrix
            Represents transitions from aggregate `a` to aggregate `b`.
        dwell_time : float
            The time spent in aggregate `a`.

        Returns
        -------
        fwd_vec : ProbabilityVector
            The product of the terms, as described above.
        """
        try:
            expQt = self.expm_calculator.compute_matrix_exp(
                        rate_matrix_aa, dwell_time)
        except:
            print rate_matrix_aa
            raise
        fwd_vec = vector_matrix_product(init_prob, expQt, do_alignment=True)
        if rate_matrix_ab is None or rate_matrix_ab.get_shape()[1] == 0:
            pass
        else:
            fwd_vec = asym_vector_matrix_product(
                        fwd_vec, rate_matrix_ab, do_alignment=True)
        return fwd_vec

    def compute_forward_vector_with_missed_events(self, init_prob,
            rate_matrix_aa, rate_matrix_ab, rate_matrix_ba, rate_matrix_bb,
            dwell_time):
        """
        Computes the product of a matrix exponential and a vector. Includes
        missed transitions from a to b (and back to a) within the dwell time.

        ``init_prob * exp(rate_matrix_aa * dwell_time) * rate_matrix_ab``

        Parameters
        ----------
        init_prob : ProbabilityVector
            The probability of starting in each state of aggregate `a`.
        rate_matrix_aa : RateMatrix
            Represents transitions within aggregate `a`.
        rate_matrix_ab : RateMatrix
            Represents transitions from aggregate `a` to aggregate `b`.
        rate_matrix_ba : RateMatrix
            Represents transitions from aggregate `b` to aggregate `a`.
        rate_matrix_bb : RateMatrix
            Represents transitions within aggregate `b`.
        dwell_time : float
            The time spent in aggregate `a`.

        Returns
        -------
        fwd_vec : ProbabilityVector
            The product of the terms, as described above.
        """
        try:
            expQt = self.expm_calculator.compute_matrix_exp(
                        rate_matrix_aa, dwell_time)
        except:
            print rate_matrix_aa
            raise
        fwd_vec = vector_matrix_product(init_prob, expQt, do_alignment=True)

        missed_events_expQt = \
            self.expm_calculator.compute_missed_events_matrix_exp(rate_matrix_aa,
                rate_matrix_ab, rate_matrix_ba, rate_matrix_bb, dwell_time,
                self.dead_time)

        fwd_vec = vector_matrix_product(fwd_vec, missed_events_expQt,
                                        do_alignment=True)

        if rate_matrix_ab is None or rate_matrix_ab.get_shape()[1] == 0:
            pass
        else:
            fwd_vec = asym_vector_matrix_product(
                        fwd_vec, rate_matrix_ab, do_alignment=True)
        return fwd_vec

    def compute_forward_state_series(self, init_prob, rate_matrix_aa,
                                     rate_matrix_ab, dwell_time, num_states=1):
        fwd_vec = self.compute_forward_vector(
                    init_prob, rate_matrix_aa, rate_matrix_ab, dwell_time)
        ml_state_series = fwd_vec.get_ml_state_series(num_states)
        return ml_state_series