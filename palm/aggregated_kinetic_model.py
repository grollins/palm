import numpy
from collections import defaultdict
from palm.base.model import Model
from palm.state_collection import StateIDCollection
from palm.rate_fcn import rate_from_rate_id
from palm.graph import make_graph_from_route_collection
from palm.rate_matrix import make_rate_matrix_from_state_ids
from palm.probability_vector import make_prob_vec_from_state_ids

class State(object):
    '''
    A generic state class for aggregated kinetic models.
    '''
    def __init__(self, id_str, observation_class):
        self.id = id_str
        self.observation_class = observation_class
        self.initial_state_flag = False

    def __str__(self):
        return "%s %s" % (self.id, self.observation_class)

    def get_id(self):
        return self.id

    def get_class(self):
        return self.observation_class

    def set_initial_state_flag(self):
        self.initial_state_flag = True

    def is_initial_state(self):
        return self.initial_state_flag

    def as_dict(self):
        return {'observation_class':self.get_class()}


class Route(object):
    '''
    A generic route class for aggregated kinetic models.
    '''
    def __init__(self, id_str, start_state_id, end_state_id, rate_id,
                 multiplicity):
        self.id = id_str
        self.start_state_id = start_state_id
        self.end_state_id = end_state_id
        self.rate_id = rate_id
        self.multiplicity = multiplicity

    def __str__(self):
        my_str = "%s %s %s %s %d" % (
                    self.id, self.start_state_id, self.end_state_id,
                    self.rate_id, self.multiplicity)
        return my_str

    def get_id(self):
        return self.id
    def get_start_state(self):
        return self.start_state_id
    def get_end_state(self):
        return self.end_state_id
    def get_multiplicity(self):
        return self.multiplicity
    def as_dict(self):
        return {'start_state':self.start_state_id,
                'end_state':self.end_state_id,
                'rate_id':self.rate_id,
                'multiplicity':self.multiplicity}


class AggregatedKineticModel(Model):
    """An AggregatedKineticModel consists of states and routes.
       The routes are transitions between states. The model is
       aggregated in the sense that each state belongs to one
       of several discrete observation classes (e.g. 'dark' or 'bright').
    """
    def __init__(self, state_enumerator, route_mapper, parameter_set):
        super(AggregatedKineticModel, self).__init__()
        self.state_enumerator = state_enumerator
        self.route_mapper = route_mapper
        self.parameter_set = parameter_set

        r = self.state_enumerator()
        self.state_collection, self.initial_state_id, self.final_state_id = r
        self.state_groups = self.state_collection.sort('observation_class')
        self.state_id_collection = self.state_collection.get_state_ids()
        self.state_ids_by_class_dict = {}
        self.state_class_by_id_dict = {}
        for obs_class, id_list in self.state_groups.groups.iteritems():
            this_state_id_collection = StateIDCollection()
            this_state_id_collection.from_state_id_list(id_list)
            self.state_ids_by_class_dict[obs_class] = this_state_id_collection
            for this_id in id_list:
                self.state_class_by_id_dict[this_id] = obs_class

        self.route_collection = self.route_mapper(self.state_collection)
        self.route_graph = make_graph_from_route_collection(
                                self.route_collection)

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

    def get_num_states(self, class_name=None):
        if class_name:
            return len(self.state_groups[class_name])
        else:
            return len(self.state_collection)

    def get_num_routes(self):
        return len(self.route_collection)

    def build_rate_matrix(self, time=0.):
        # need a state id collection
        rate_matrix = make_rate_matrix_from_state_ids(self.state_id_collection)
        for r_id, r in self.route_collection.iter_routes():
            start_id = r['start_state']
            end_id = r['end_state']
            rate_id = r['rate_id']
            multiplicity = r['multiplicity']
            this_rate = multiplicity * rate_from_rate_id(rate_id, time,
                                                         self.parameter_set)
            assert this_rate >= 0.0, "%s %s %.2e" % (start_id, end_id, this_rate)
            rate_matrix.set_rate(start_id, end_id, this_rate)
        rate_matrix.balance_transition_rates()
        return rate_matrix

    def get_submatrix(self, rate_matrix, start_class, end_class):
        start_id_collection = self.state_ids_by_class_dict[start_class]
        end_id_collection = self.state_ids_by_class_dict[end_class]
        submatrix = rate_matrix.get_submatrix(
                        start_id_collection, end_id_collection)
        return submatrix

    def get_local_matrix(self, time, start_state_series, depth):
        # find routes to neighbor states
        # build a rate matrix from neighbor routes and states
        r_list = []
        local_state_id_list = []
        for start_state_id in start_state_series.index:
            for r_id in self.map_local_routes(start_state_id, depth):
                r = self.route_collection.get_route_from_id_as_series(r_id)
                r_list.append(r)
                start_id = r['start_state']
                end_id = r['end_state']
                if start_id in local_state_id_list:
                    pass
                else:
                    local_state_id_list.append(start_id)
                if end_id in local_state_id_list:
                    pass
                else:
                    local_state_id_list.append(end_id)
        if len(local_state_id_list) == 0:
            # at the end, when only the fully photobleached state remains,
            # this line is necessary to avoid an empty rate matrix
            local_state_id_list += start_state_series.index.tolist()
        local_state_id_collection = StateIDCollection()
        local_state_id_collection.from_state_id_list(local_state_id_list)
        rate_matrix = make_rate_matrix_from_state_ids(
                        index_id_collection=local_state_id_collection,
                        column_id_collection=local_state_id_collection)
        for r in r_list:
            start_id = r['start_state']
            end_id = r['end_state']
            rate_id = r['rate_id']
            multiplicity = r['multiplicity']
            this_rate = multiplicity * rate_from_rate_id(
                                            rate_id, time, self.parameter_set)
            assert this_rate >= 0.0, "%s %s %.2e" % (start_id, end_id, this_rate)
            rate_matrix.set_rate(start_id, end_id, this_rate)
        rate_matrix.balance_transition_rates()
        return rate_matrix

    def get_local_submatrix(self, rate_matrix, start_class, end_class):
        id_list = rate_matrix.get_index_id_list()
        if len(id_list) == 1 and id_list[0] == self.final_state_id:
            start_id_list = end_id_list = id_list
        else:
            start_id_list = []
            end_id_list = []
            for this_id in id_list:
                this_obs_class = self.state_class_by_id_dict[this_id]
                if this_obs_class == start_class:
                    start_id_list.append(this_id)
                if this_obs_class == end_class:
                    end_id_list.append(this_id)
        start_id_collection = StateIDCollection()
        start_id_collection.from_state_id_list(start_id_list)
        end_id_collection = StateIDCollection()
        end_id_collection.from_state_id_list(end_id_list)
        submatrix = rate_matrix.get_submatrix(
                        start_id_collection, end_id_collection)
        return submatrix

    def map_local_routes(self, start_state_id, depth):
        for route_id in self.route_graph.iter_successors(start_state_id, depth):
            yield route_id

    def get_local_vec(self, rate_matrix, start_state_series):
        index_list = rate_matrix.get_index_id_list()
        state_id_collection = StateIDCollection()
        state_id_collection.from_state_id_list(index_list)
        local_vec = make_prob_vec_from_state_ids(state_id_collection)
        for start_state_id in start_state_series.index:
            state_prob = start_state_series[start_state_id]
            try:
                local_vec.set_state_probability(start_state_id, state_prob)
            except KeyError:
                print local_vec, rate_matrix
                raise
        return local_vec
