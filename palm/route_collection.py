import pandas
from .state_collection import StateIDCollection


class RouteCollectionFactory(object):
    """docstring for RouteCollectionFactory"""
    def __init__(self):
        super(RouteCollectionFactory, self).__init__()
        self.route_list = []
        self.route_id_list = []
    def add_route(self, route):
        self.route_id_list.append( route.get_id() )
        self.route_list.append( route.as_dict() )
    def make_route_collection(self):
        route_collection = RouteCollection()
        route_collection.data_frame = pandas.DataFrame(
                                        self.route_list,
                                        index=self.route_id_list)
        return route_collection


class RouteCollection(object):
    """docstring for RouteCollection"""
    def __init__(self):
        super(RouteCollection, self).__init__()
        self.data_frame = None
    def __len__(self):
        return len(self.data_frame)
    def __str__(self):
        return self.data_frame.to_string()
    def iter_routes(self):
        for route_id, route_series in self.data_frame.iterrows():
            yield (route_id, route_series)
    def get_route_ids(self):
        s = RouteIDCollection()
        s.route_id_list = self.data_frame.index.tolist()
        return s
    def get_start_state_series(self):
        return self.data_frame['start_state']
    def get_end_state_series(self):
        return self.data_frame['end_state']
    def sort(self, sort_column):
        return self.data_frame.groupby(sort_column)
    def get_subset_from_id_collection(self, r_id_collection):
        sub_collection = RouteCollection()
        # sub_df = self.data_frame.ix[r_id_collection.as_list()]
        sub_df = self.data_frame.reindex(r_id_collection.as_list())
        sub_collection.data_frame = sub_df
        return sub_collection
    def get_unique_state_ids(self):
        local_state_id_collection = StateIDCollection()
        start_state_ids = self.get_start_state_series()
        end_state_ids = self.get_end_state_series()
        local_state_ids = start_state_ids.append(end_state_ids)
        unique_local_state_ids = local_state_ids.drop_duplicates()
        local_state_id_collection.add_state_id_list(
                                    unique_local_state_ids.tolist())
        return local_state_id_collection


class RouteIDCollection(object):
    """docstring for RouteIDCollection"""
    def __init__(self):
        super(RouteIDCollection, self).__init__()
        self.route_id_list = []
    def __str__(self):
        return str(self.route_id_list)
    def __iter__(self):
        for r in self.route_id_list:
            yield r
    def __contains__(self, route_id):
        return (route_id in self.route_id_list)
    def __len__(self):
        return len(self.route_id_list)
    def add_id(self, route_id):
        self.route_id_list.append(route_id)
    def from_route_id_list(self, route_id_list):
        self.route_id_list = route_id_list
    def from_route_collection(self, route_collection):
        self.route_id_list = route_collection.get_id_list()
    def as_list(self):
        return self.route_id_list
