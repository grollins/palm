import numpy
import cPickle
from collections import defaultdict

class ParameterSetDistribution(object):
    """docstring for ParameterSetDistribution"""
    def __init__(self):
        super(ParameterSetDistribution, self).__init__()
        self.distribution = defaultdict(list)

    def __eq__(self, other_parameter_set):
        parameter_name = self.distribution.keys()[0]
        parameter_dist = self.single_parameter_distribution_as_array(parameter_name)
        other_parameter_dist = other_parameter_set.single_parameter_distribution_as_array(parameter_name)
        is_equal = True
        for i in xrange(len(parameter_dist)):
            if parameter_dist[i] == other_parameter_dist[i]:
                continue
            else:
                is_equal = False
                break
        return is_equal

    def add_parameter_set(self, parameter_set, score):
        for param_name, param_value in parameter_set:
            self.distribution[param_name].append(param_value)
            self.distribution['score'].append(score)

    def single_parameter_distribution_as_array(self, parameter_name):
        return numpy.array(self.distribution[parameter_name])

    def save_to_file(self, filename):
        output_stream = open(filename, 'wb')
        cPickle.dump(self.distribution, output_stream)
        output_stream.close()

    def load_from_file(self, filename):
        input_stream = open(filename, 'rb')
        self.distribution = cPickle.load(input_stream)
        input_stream.close()
