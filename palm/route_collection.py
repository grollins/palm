import pandas

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
    def sort(self, sort_column):
        return self.data_frame.groupby(sort_column)
    def get_route_from_id_as_series(self, r_id):
        return self.data_frame.ix[r_id]
