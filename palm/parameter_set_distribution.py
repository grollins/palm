import numpy
import pandas
from collections import defaultdict

class ParamSetDistFactory(object):
    def __init__(self):
        super(ParamSetDistFactory, self).__init__()
        self.distribution_dict = defaultdict(list)
        self.param_set_list = []

    def add_parameter_set(self, parameter_set):
        for param_name, param_value in parameter_set:
            self.add_parameter(param_name, param_value)

    def add_parameters_from_data_series(self, ds):
        for param_name in ds.keys():
            param_value = ds[param_name]
            self.add_parameter(param_name, param_value)

    def add_parameter(self, parameter_name, parameter_value):
        self.distribution_dict[parameter_name].append(parameter_value)

    def make_psd(self):
        return ParameterSetDistribution(self.distribution_dict)


class ParameterSetDistribution(object):
    """
    A distribution of parameter values, probably from
    repeated bootstrap fittings of data.
    """
    def __init__(self, dist_dict=None):
        if dist_dict:
            self.data_frame = pandas.DataFrame(dist_dict)
        else:
            self.data_frame = None

    def __len__(self):
        return len(self.data_frame)

    def __str__(self):
        return str(self.data_frame)

    def __iter__(self):
        for i, row in self.data_frame.T.iteritems():
            yield i, row

    def single_parameter_distribution_as_array(self, parameter_name):
        return numpy.array(self.data_frame[parameter_name])

    def select_param_sets(self, parameter_name, parameter_value):
        condition = self.data_frame[parameter_name] == parameter_value
        selected_data = self.data_frame[condition]
        return selected_data

    def save_to_file(self, filename):
        self.data_frame.save(filename)

    def load_from_file(self, filename, append_data=False):
        loaded_data_frame = pandas.load(filename)
        if append_data:
            self.data_frame = pandas.concat([self.data_frame,
                                             loaded_data_frame])
        else:
            self.data_frame = pandas.load(filename)

    def sort_index(self, sort_by_column, is_ascending=True):
        self.data_frame = self.data_frame.sort_index(by=sort_by_column,
                                                     ascending=is_ascending)

    def to_html(self, filename):
        html_str = self.data_frame.to_html()
        with open(filename, 'w') as f:
            f.write(html_str)
        