import pandas

class StateCollectionFactory(object):
    """docstring for StateCollectionFactory"""
    def __init__(self):
        super(StateCollectionFactory, self).__init__()
        self.state_list = []
        self.state_id_list = []
    def add_state(self, state):
        self.state_id_list.append( state.get_id() )
        self.state_list.append( state.as_dict() )
    def make_state_collection(self):
        state_collection = StateCollection()
        state_collection.data_frame = pandas.DataFrame(
                                        self.state_list,
                                        index=self.state_id_list)
        return state_collection


class StateCollection(object):
    """docstring for StateCollection"""
    def __init__(self):
        super(StateCollection, self).__init__()
        self.data_frame = None
    def __len__(self):
        return len(self.data_frame)
    def __str__(self):
        return self.data_frame.to_string()
    def iter_states(self):
        for state_id, state_series in self.data_frame.iterrows():
            yield (state_id, state_series)
    def get_state_ids(self):
        s = StateIDCollection()
        s.state_id_list = self.data_frame.index.tolist()
        return s
    def sort(self, sort_column):
        return self.data_frame.groupby(sort_column)

class StateIDCollection(object):
    """docstring for StateIDCollection"""
    def __init__(self):
        super(StateIDCollection, self).__init__()
        self.state_id_list = []
    def __str__(self):
        return str(self.state_id_list)
    def __iter__(self):
        for s in self.state_id_list:
            yield s
    def __contains__(self, state_id):
        return (state_id in self.state_id_list)
    def __len__(self):
        return len(self.state_id_list)
    def add_id(self, state_id):
        self.state_id_list.append(state_id)
    def from_state_id_list(self, state_id_list):
        self.state_id_list = state_id_list
    def from_state_collection(self, state_collection):
        self.state_id_list = state_collection.get_id_list()
    def as_list(self):
        return self.state_id_list
