import numpy
import pandas

def make_prob_vec_from_state_ids(state_id_collection):
    pv = ProbabilityVector()
    pv.series = pandas.Series(0.0, index=state_id_collection.as_list())
    return pv

def make_prob_vec_from_panda_series(series):
    pv = ProbabilityVector()
    pv.series = series
    return pv

class ProbabilityVector(object):
    """docstring for ProbabilityVector"""
    def __init__(self):
        super(ProbabilityVector, self).__init__()
        self.series = None
    def __len__(self):
        return len(self.series)
    def __str__(self):
        return str(self.series)
    def set_state_probability(self, state_id, probability):
        self.series[state_id] = probability
    def get_state_probability(self, state_id):
        return self.series[state_id]
    def set_uniform_state_probability(self):
        self.series[:] = 1./len(self)
    def sum_vector(self):
        return self.series.sum()
    def scale_vector(self, scale_factor):
        self.series *= scale_factor
    def get_ml_state_series(self, num_states, threshold=0.0):
        above_threshold = self.series[self.series > threshold]
        ordered_series = above_threshold.order(ascending=False)
        upper_limit = min(num_states, len(ordered_series))
        return ordered_series[:upper_limit]
    def combine_first(self, vec):
        # self clobbers vec
        return self.series.combine_first(vec.series)


class VectorTrajectory(object):
    """docstring for VectorTrajectory"""
    def __init__(self, state_id_collection):
        super(VectorTrajectory, self).__init__()
        self.state_id_collection = state_id_collection
        self.vec_list = []
    def __len__(self):
        return len(self.vec_list)
    def __str__(self):
        full_str = ""
        for v in iter(self):
            full_str += str(v)
            full_str += "\n"
        return full_str
    def __iter__(self):
        for v in iter(self.vec_list):
            yield v
    def add_vector(self, vec):
        vec_template = make_prob_vec_from_state_ids(self.state_id_collection)
        combined_vec = vec.combine_first(vec_template)
        self.vec_list.append(combined_vec)
