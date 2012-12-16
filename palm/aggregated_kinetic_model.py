import numpy
import base.model
from collections import defaultdict
from rate_matrix import RateMatrixFactory, AggregatedRateMatrix

class State(object):
    def __init__(self, id_str, observation_class):
        self.id = id_str
        self.observation_class = observation_class

    def __str__(self):
        return "%s %s" % (self.id, self.observation_class)

    def get_id(self):
        return self.id

    def get_class(self):
        return self.observation_class


class Route(object):
    def __init__(self, start_state, end_state, log_rate_function):
        self.start_state = start_state
        self.end_state = end_state
        self.log_rate_function = log_rate_function

    def __str__(self):
        my_str = "%s %s %.3e" % (self.start_state, self.end_state,
                                 self.log_rate_function(t=0.0))
        return my_str

    def compute_log_rate(self, t):
        return self.log_rate_function(t)


class AggregatedKineticModel(base.model.Model):
    """docstring for AggregatedKineticModel"""
    def __init__(self, state_enumerator, route_mapper, parameter_set):
        super(AggregatedKineticModel, self).__init__()
        self.state_enumerator = state_enumerator
        self.route_mapper = route_mapper
        self.parameter_set = parameter_set
        self.rate_matrix_factory = RateMatrixFactory(AggregatedRateMatrix)

        unsorted_state_list = self.state_enumerator()
        sorting_results = self._sort_states_by_observation_class(unsorted_state_list)
        self.states = sorting_results[0]
        self.state_index_dict = sorting_results[1]
        self.class_indices_dict = sorting_results[2]

        self.routes = self.route_mapper(self.states, self.state_index_dict)
        self.rate_matrix = None

    def _sort_states_by_observation_class(self, unsorted_state_list):
        # initialize data structures for state sorting
        unsorted_state_index_dict = {}
        unsorted_class_indices_dict = defaultdict(list)
        sorted_state_list = []
        sorted_state_index_dict = {}
        sorted_class_indices_dict = defaultdict(list)

        # store current state indices by class name and by state id
        for this_index, this_state in enumerate(unsorted_state_list):
            unsorted_state_index_dict[this_state.get_id()] = this_index
            unsorted_class_indices_dict[this_state.get_class()].append(this_index)

        # sort states by class name
        sorted_index = 0
        for key, value in unsorted_class_indices_dict.iteritems():
            this_class_indices_list = value
            sorted_indices_list = []
            for index in this_class_indices_list:
                this_state = unsorted_state_list[index]
                sorted_state_list.append(this_state)
                sorted_indices_list.append(sorted_index)
                sorted_index += 1
            sorted_class_indices_dict[key] = sorted_indices_list

        # store sorted state indices by class name and by state id
        for this_index, this_state in enumerate(sorted_state_list):
            sorted_state_index_dict[this_state.get_id()] = this_index

        # return sorted states
        return sorted_state_list, sorted_state_index_dict,\
               sorted_class_indices_dict

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

    def get_num_states(self, class_name=None):
        if class_name:
            return len(self.class_indices_dict[class_name])
        else:
            return len(self.states)

    def get_num_routes(self):
        return len(self.routes)

    def build_rate_matrix(self, time=0.):
        rate_matrix = self.rate_matrix_factory.create_rate_matrix(self.get_num_states(),
                                                                  self.routes,
                                                                  self.class_indices_dict,
                                                                  time)
        self.rate_matrix = rate_matrix

    def get_numpy_submatrix(self, start_class, end_class):
        return self.rate_matrix.get_numpy_submatrix(start_class, end_class)
