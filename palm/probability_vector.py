import numpy
import scipy.linalg
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
    def fill_zeros(self, value):
        self.series[self.series == 0.0] = value
    def fill_na(self, value):
        self.series = self.series.fillna(value)
    def fill_negative(self, value):
        self.series[self.series < 0.0] = value
    def fill_positive(self, value):
        self.series[self.series > 1.0] = value
    def as_npy_array(self):
        return numpy.array(self.series)
    def allclose(self, other_vec):
        return numpy.allclose(self.series.values,
                              other_vec.series.values)
    def compute_norm(self):
        return scipy.linalg.norm(self.series)


class VectorTrajectory(object):
    """docstring for VectorTrajectory"""
    def __init__(self, state_id_list):
        super(VectorTrajectory, self).__init__()
        self.state_id_list = state_id_list
        self.time_list = []
        self.vec_list = []
    def __len__(self):
        return len(self.vec_list)
    def __str__(self):
        full_str = ""
        for t,v in iter(self):
            full_str += "%.2e\n%s\n" % (t,str(v))
        return full_str
    def __iter__(self):
        for t,v in zip(self.time_list, self.vec_list):
            yield t,v
    def add_vector(self, time, vec):
        self.time_list.append(time)
        vec_template = make_prob_vec_from_state_ids(self.state_id_list)
        combined_vec_series = vec.combine_first(vec_template)
        combined_vec = make_prob_vec_from_panda_series(combined_vec_series)
        self.vec_list.append(combined_vec)
    def convert_to_df(self):
        vec_dict_list = []
        for t,v in self:
            vec_dict_list.append( v.series.to_dict() )
        df = pandas.DataFrame(vec_dict_list)
        df['time'] = pandas.Series(self.time_list)
        return df
