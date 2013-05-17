from palm.linalg import ScipyMatrixExponential
from palm.linalg import vector_matrix_product
from palm.probability_vector import VectorTrajectory
from palm.probability_vector import make_prob_vec_from_state_ids

class Dynamics(object):
    """docstring for Dynamics"""
    def __init__(self, noisy=False):
        self.expm = ScipyMatrixExponential()
        self.noisy = noisy

    def compute_trajectory(self, model, time_array):
        # init_prob_vec = model.get_initial_probability_vector()
        init_prob_vec = make_prob_vec_from_state_ids(model.state_id_collection)
        init_prob_vec.set_state_probability(model.all_inactive_state_id, 1.0)
        Q = model.build_rate_matrix(0.0)

        if self.noisy:
            print init_prob_vec
            print Q

        vec_traj = VectorTrajectory(model.state_id_collection)
        for i,t in enumerate(time_array):
            try:
                eQt = self.expm.compute_matrix_exp(Q, t)
            except ValueError:
                print "eQt not finite at %d, %.2e" % (i, t)
                print Q
                break
            prob_vec_at_time_t = vector_matrix_product(
                                    init_prob_vec, eQt,
                                    do_alignment=True)
            vec_traj.add_vector(t, prob_vec_at_time_t)
        return vec_traj
