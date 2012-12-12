import numpy
import scipy.linalg

ALMOST_ZERO = 1e-200

def spectral_decomposition(rate_matrix):
    vals, left, right = scipy.linalg.eig(rate_matrix, left=True, right=True)
    vals = vals.real
    left_inv = scipy.linalg.inv(left)
    right_inv = scipy.linalg.inv(right)
    A_list = []
    for i in xrange(left.shape[0]):
        left_col = left[:,i:i+1]
        left_inv_row = left_inv[i:i+1,:]
        A = numpy.dot(left_col, left_inv_row)
        A_list.append(numpy.atleast_2d(A))
    return A_list, vals

def compute_gamma(i, j, t, eig_vals):
    lambda_i = eig_vals[i]
    lambda_j = eig_vals[j]

    if i == j:
        gamma_ij = t * numpy.exp(lambda_i * t)
    else:
        numerator = numpy.exp(lambda_i * t) - numpy.exp(lambda_j * t)
        denominator = lambda_i - lambda_j
        gamma_ij = numerator / denominator

    return gamma_ij

class VectorSet(object):
    def __init__(self):
        self.vector_dict = {}
    def add_vector(self, key, vec):
        assert vec.shape[1] == 1
        self.vector_dict[key] = vec
    def get_vector(self, key):
        return self.vector_dict[key]

class ScalingCoefficients(object):
    def __init__(self):
        self.coef_dict = {}
    def __len__(self):
        return len(self.coef_dict)
    def add_coef(self, key, coef):
        self.coef_dict[key] = coef
    def get_coef(self, key):
        return self.coef_dict[key]
    def compute_product(self):
        c_array = numpy.array(self.coef_dict.values())
        return numpy.prod(c_array)

class SachsAggregatedMarkov(object):
    '''
    Based on Qin, Auerbach, and Sachs,
    Proc. R. Soc. Lond. B (1997) 264, 375-383.
    '''
    def __init__(self, trajectory, model, noisy=False):
        self.L = len(trajectory)
        self.trajectory = trajectory
        self.model = model
        self.noisy = noisy

        init_class = self.trajectory.get_dwell_class(1)
        self.init_pop = self.model.get_init_pop(init_class)

    def compute_forward_vectors(self, scaling=True):
        '''
        Eqns 9 and 10 in Qin et al
        '''
        alpha_set = VectorSet()
        c_set = ScalingCoefficients()
        for k in xrange(self.L+1):
            if k == 0:
                alpha_0_T = self.init_pop # assuming init_pop is a row vector
                assert type(alpha_0_T) is numpy.matrix
                c_0 = 1./numpy.sum(alpha_0_T)
                c_set.add_coef(k, c_0)
                scaled_alpha_0_T = alpha_0_T * c_0
                if scaling:
                    scaled_alpha_0 = scaled_alpha_0_T.T
                    alpha_set.add_vector(k, scaled_alpha_0)
                    prev_alpha_T = scaled_alpha_0_T
                else:
                    alpha_0 = alpha_0_T.T
                    alpha_set.add_vector(k, alpha_0)
                    prev_alpha_T = alpha_0_T
            else:
                this_time = self.trajectory.get_cumulative_time(k)
                self.model.build_rate_matrix(time=this_time)
                assert type(self.model.rate_matrix) is numpy.matrix
                t_k = self.trajectory.get_dwell_time(k)
                start_class = self.trajectory.get_dwell_class(k)
                end_class = self.trajectory.get_dwell_class(k+1)
                G = self.get_G_matrix(t_k, start_class, end_class)
                assert type(G) is numpy.matrix
                # alpha_k_T = numpy.dot(prev_alpha_T, G)
                alpha_k_T = prev_alpha_T * G
                assert type(alpha_k_T) is numpy.matrix
                inv_c = numpy.sum(alpha_k_T)
                if inv_c < ALMOST_ZERO:
                    # print "Underflow in ml.sachs.compute_forward_vectors"
                    inv_c = ALMOST_ZERO
                c_k = 1./inv_c
                # if self.noisy:
                #     print k, c_k
                #     print numpy.sum(alpha_k_T)
                c_set.add_coef(k, c_k)
                scaled_alpha_k_T = alpha_k_T * c_k
                if scaling:
                    scaled_alpha_k = scaled_alpha_k_T.T
                    alpha_set.add_vector(k, scaled_alpha_k)
                    prev_alpha_T = scaled_alpha_k_T
                else:
                    alpha_k = alpha_k_T.T
                    alpha_set.add_vector(k, alpha_k)
                    prev_alpha_T = alpha_k_T
        return alpha_set, c_set

    def compute_backward_vectors(self, c_set, scaling=True):
        '''
        Eqns 12 and 13 in Qin et al
        '''
        beta_set = VectorSet()
        # k in (L+1, L, ... , 1)
        for k in xrange(self.L+1, 0, -1):
            if k == (self.L+1):
                ones_column_vector = numpy.ones([self.init_pop.shape[1],1])
                beta_set.add_vector(k, ones_column_vector)
                next_beta = ones_column_vector
            else:
                this_time = self.trajectory.get_cumulative_time(k)
                self.model.build_rate_matrix(time=this_time)
                t_k = self.trajectory.get_dwell_time(k)
                start_class = self.trajectory.get_dwell_class(k)
                end_class = self.trajectory.get_dwell_class(k+1)
                G = self.get_G_matrix(t_k, start_class, end_class)
                # print G.shape, next_beta.shape
                beta_k = numpy.dot(G, next_beta)
                scaled_beta_k = c_set.get_coef(k) * beta_k
                if scaling:
                    beta_set.add_vector(k, scaled_beta_k)
                    next_beta = scaled_beta_k
                else:
                    beta_set.add_vector(k, beta_k)
                    next_beta = beta_k
        return beta_set

    def get_G_matrix(self, t_k, start_class, end_class):
        '''
        Eqn 4 in Qin et al
        '''
        Q_aa = self.model.get_submatrix(start_class, start_class)
        Q_ab = self.model.get_submatrix(start_class, end_class)
        assert type(Q_aa) is numpy.matrix
        assert type(Q_ab) is numpy.matrix
        # A_list, eig_vals = spectral_decomposition(Q_aa)
        # A_array = numpy.array(A_list)
        # A_array = numpy.zeros([A_list[0].shape[0], A_list[0].shape[1], len(A_list)])
        # for i,a in enumerate(A_list):
        #     A_array[:,:,i] = a
        # A_sum = numpy.zeros_like(A_array[:,:,0])
        # A_sum = numpy.sum(A_array * numpy.exp(eig_vals * t_k), axis=2)
        # G = numpy.dot(A_sum, Q_ab)
        # G = numpy.dot( scipy.linalg.expm(Q_aa * t_k), Q_ab )
        # if self.noisy:
        #     print Q_aa * t_k
        #     print Q_ab
        G = scipy.linalg.expm(Q_aa * t_k) * Q_ab
        return G

    def compute_likelihood(self, total_time, scaling=True):
        '''
        Eqn 11 in Qin et al
        '''
        alpha_set, c_set = self.compute_forward_vectors(scaling)
        if self.noisy:
            print self.trajectory
            for i in xrange(len(c_set)):
                this_coef = c_set.get_coef(i)
                this_vec = alpha_set.get_vector(i)
                print i, this_coef, this_vec
        assert len(c_set) == self.L+1

        last_alpha = alpha_set.get_vector(self.L)
        cumulative_time = self.trajectory.get_cumulative_time(self.L)
        remaining_time = total_time - cumulative_time
        if remaining_time < 0.:
            print "Warning: trajectory longer than expected time of %.1f" % total_time
        Q_dd = self.model.get_submatrix("dark", "dark")
        assert type(Q_dd) is numpy.matrix
        try:
            last_alpha_T = last_alpha.T * scipy.linalg.expm(Q_dd * remaining_time)
        except ValueError:
            print self.trajectory
            assert False

        if scaling:
            last_c = 1./numpy.sum(last_alpha_T)
            likelihood = 1./(c_set.compute_product() * last_c)
            # log_likelihood = -numpy.log10(c_set.compute_product())
        else:
            likelihood = numpy.sum(last_alpha_T)
            # log_likelihood = numpy.log10(numpy.sum(last_alpha))
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
        return likelihood

    def compute_llhood_ab_derivative(self, a_class, b_class, scaling=True):
        '''
        Eqns 27 & 28 in Qin et al
        '''
        alpha_set, c_set = self.compute_forward_vectors(scaling=scaling)
        beta_set = self.compute_backward_vectors(c_set, scaling=scaling)
        Q_aa = self.model.get_submatrix(a_class, a_class)
        A_list, eig_vals = spectral_decomposition(Q_aa)
        n_a = len(self.model.get_class_inds(a_class))
        n_b = len(self.model.get_class_inds(b_class))
        for i in xrange(n_a):
            # print i
            outer_sum = 0.0
            for k in xrange(1, self.L+1):
                inner_sum = numpy.zeros([n_a, n_b])
                this_time = self.trajectory.get_cumulative_time(k)
                self.model.build_rate_matrix(time=this_time)
                t_k = self.trajectory.get_dwell_time(k)
                start_class = self.trajectory.get_dwell_class(k)
                end_class = self.trajectory.get_dwell_class(k+1)
                if start_class == a_class and end_class == b_class:
                    # print k
                    # print alpha_list[k-2].shape, beta_list[k-1].shape
                    this_alpha = alpha_set.get_vector(k-1)
                    this_beta = beta_set.get_vector(k+1)
                    inner_sum += numpy.dot(this_alpha, this_beta.T) * numpy.exp(eig_vals[i] * t_k)
            inner_sum = numpy.atleast_2d(inner_sum)
            # print A_list[i].T.shape, inner_sum.shape
            outer_sum += numpy.dot(A_list[i].T, inner_sum)
            # print outer_sum
        return outer_sum

    def compute_llhood_aa_derivative(self, a_class, scaling=True):
        '''
        Eqns 27 & 28 in Qin et al
        '''
        alpha_set, c_set = self.compute_forward_vectors(scaling=scaling)
        beta_set = self.compute_backward_vectors(c_set, scaling=scaling)
        Q_aa = self.model.get_submatrix(a_class, a_class)
        A_list, eig_vals = spectral_decomposition(Q_aa)
        n_a = len(self.model.get_class_inds(a_class))
        for i in xrange(n_a):
            outer_sum = 0.0
            for j in xrange(n_a):
                for k in xrange(1, self.L+1):
                    inner_sum = numpy.zeros([n_a, n_a])
                    this_time = self.trajectory.get_cumulative_time(k)
                    self.model.build_rate_matrix(time=this_time)
                    t_k = self.trajectory.get_dwell_time(k)
                    start_class = self.trajectory.get_dwell_class(k)
                    end_class = self.trajectory.get_dwell_class(k+1)
                    Q_a1a2 = self.model.get_submatrix(start_class, end_class)
                    if start_class == a_class:
                        # print k
                        this_alpha = alpha_set.get_vector(k-1)
                        this_beta = beta_set.get_vector(k+1)
                        ab = numpy.dot(this_alpha, this_beta.T)
                        gamma_ij = compute_gamma(i, j, t_k, eig_vals)
                        inner_sum += numpy.dot(ab, Q_a1a2.T) * gamma_ij
                        # inner_sum += ab * gamma_ij
                partial_product = numpy.dot(A_list[i].T, inner_sum)
                # print partial_product.shape, A_list[j].T.shape
                outer_sum += numpy.dot( partial_product, A_list[j].T )
        return outer_sum
