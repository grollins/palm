import numpy
from collections import defaultdict
from palm.base.model import Model
from palm.state_collection import StateIDCollection
from palm.route_collection import RouteIDCollection
from palm.rate_fcn import rate_from_rate_id
from palm.graph import make_graph_from_route_collection
from palm.rate_matrix import make_rate_matrix_from_state_ids
from palm.probability_vector import make_prob_vec_from_state_ids


class AggregatedKineticModel(Model):
    """
    An AggregatedKineticModel consists of states and routes.
    The routes are transitions between states. The model is
    aggregated in the sense that each state belongs to one
    of several discrete aggregated classes (e.g. 'dark' or 'bright').
    Note that the term `class` in "aggregated class" does not refer
    to the python concept of a class; it's a different meaning.

    Parameters
    ----------
    state_enumerator : callable f()
        Generates a StateCollection for the model.
    route_mapper : callable f(state_collection)
        Generates a RouteCollection for the model.
    parameter_set : ParameterSet
    fermi_activation : bool, optional
        Whether the activation rates vary with time.

    Attributes
    ----------
    state_collection : StateCollection
    state_groups : pandas.DataFrame
    state_id_collection : StateIDCollection
    state_ids_by_class_dict : dict
        Lists of state ids, indexed by class name.
    state_class_by_id_dict : dict
        Aggregated class of each state, indexed by state id.
    route_collection : RouteCollection
    route_graph : DiGraph
        Work in progress. Would only be used for local matrix methods.

    """
    def __init__(self, state_enumerator, route_mapper, parameter_set,
                 fermi_activation=False):
        super(AggregatedKineticModel, self).__init__()
        self.state_enumerator = state_enumerator
        self.route_mapper = route_mapper
        self.parameter_set = parameter_set
        self.fermi_activation = fermi_activation

        r = self.state_enumerator()
        self.state_collection, self.initial_state_id, self.final_state_id = r
        self.state_groups = self.state_collection.sort('observation_class')
        self.state_id_collection = self.state_collection.get_state_ids()
        self.state_ids_by_class_dict = {}
        self.state_class_by_id_dict = {}
        for obs_class, id_list in self.state_groups.groups.iteritems():
            this_state_id_collection = StateIDCollection()
            this_state_id_collection.add_state_id_list(id_list)
            self.state_ids_by_class_dict[obs_class] = this_state_id_collection
            for this_id in id_list:
                self.state_class_by_id_dict[this_id] = obs_class

        self.route_collection = self.route_mapper(self.state_collection)
        # self.route_graph = make_graph_from_route_collection(
        #                         self.route_collection)

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

    def get_num_states(self, class_name=None):
        if class_name:
            return len(self.state_ids_by_class_dict[class_name])
        else:
            return len(self.state_id_collection)

    def get_num_routes(self):
        return len(self.route_collection)

    def build_rate_matrix(self, time=0.):
        """
        Returns
        -------
        rate_matrix : RateMatrix
        """
        rate_matrix = self._build_rate_matrix_from_routes(
                            self.state_id_collection, self.route_collection,
                            time)
        return rate_matrix

    def get_submatrix(self, rate_matrix, start_class, end_class):
        """
        Returns
        -------
        submatrix : RateMatrix
        """
        start_id_collection = self.state_ids_by_class_dict[start_class]
        end_id_collection = self.state_ids_by_class_dict[end_class]
        submatrix = rate_matrix.get_submatrix(
                        start_id_collection, end_id_collection)
        return submatrix

    def get_local_matrix(self, time, start_state_series, depth):
        # find routes to neighbor states
        r = self._find_routes_to_local_neighbors(start_state_series, depth)
        local_state_id_collection, local_routes = r

        # build a rate matrix from neighbor routes and states
        rate_matrix = self._build_rate_matrix_from_routes(
                        local_state_id_collection, local_routes, time)
        return rate_matrix

    def _find_routes_to_local_neighbors(self, start_state_series, depth):
        local_routes = self._local_routes_from_graph(start_state_series, depth)
        local_state_id_collection = local_routes.get_unique_state_ids()
        if len(local_state_id_collection) == 0:
            # at the end, when only the fully photobleached state remains,
            # this line is necessary to avoid an empty rate matrix
            local_state_id_collection.add_state_id_list(
                                    start_state_series.index.tolist())
        else:
            pass
        return local_state_id_collection, local_routes

    def _local_routes_from_graph(self, start_state_series, depth):
        r_id_collection = RouteIDCollection()
        for start_state_id in start_state_series.index:
            for r_id in self.route_graph.iter_successors(
                            start_state_id, depth):
                r_id_collection.add_id(r_id)
        local_routes = self.route_collection.get_subset_from_id_collection(
                        r_id_collection)
        return local_routes

    def _build_rate_matrix_from_routes(self, state_id_collection, routes, time):
        """
        Parameters
        ----------
        state_id_collection : StateIDCollection
        routes : RouteCollection
        time : float
            Cumulative time since start of trajectory,
            needed to compute time-dependent rates.

        Returns
        -------
        rate_matrix : RateMatrix
        """
        rate_matrix = make_rate_matrix_from_state_ids(
                        index_id_collection=state_id_collection,
                        column_id_collection=state_id_collection)
        for r_id, r in routes.iter_routes():
            start_id = r['start_state']
            end_id = r['end_state']
            rate_id = r['rate_id']
            multiplicity = r['multiplicity']
            this_rate = multiplicity * rate_from_rate_id(
                                            rate_id, time, self.parameter_set,
                                            self.fermi_activation)
            rate_matrix.set_rate(start_id, end_id, this_rate)
        rate_matrix.balance_transition_rates()
        return rate_matrix

    def get_local_submatrix(self, rate_matrix, start_class, end_class):
        id_list = rate_matrix.get_index_id_list()
        if len(id_list) == 1 and id_list[0] == self.final_state_id:
            # edge case where only the photobleached state is left
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
        start_id_collection.add_state_id_list(start_id_list)
        end_id_collection = StateIDCollection()
        end_id_collection.add_state_id_list(end_id_list)
        submatrix = rate_matrix.get_submatrix(
                        start_id_collection, end_id_collection)
        return submatrix

    def get_local_vec(self, rate_matrix, start_state_series):
        index_list = rate_matrix.get_index_id_list()
        state_id_collection = StateIDCollection()
        state_id_collection.add_state_id_list(index_list)
        local_vec = make_prob_vec_from_state_ids(state_id_collection)
        for start_state_id in start_state_series.index:
            state_prob = start_state_series[start_state_id]
            try:
                local_vec.set_state_probability(start_state_id, state_prob)
            except KeyError:
                print local_vec, rate_matrix
                raise
        return local_vec
