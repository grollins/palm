import pandas

class StateCollectionFactory(object):
    """
    Factory class that builds a StateCollection.

    Attributes
    ----------
    state_list : list
        The states in the model.
    state_id_list : list
        The identifier strings for each state.
    """
    def __init__(self):
        super(StateCollectionFactory, self).__init__()
        self.state_list = []
        self.state_id_list = []

    def add_state(self, state):
        """
        Adds a state (in dict form) to the growing list of states.

        Parameters
        ----------
        state : State
        """
        self.state_id_list.append( state.get_id() )
        self.state_list.append( state.as_dict() )

    def make_state_collection(self):
        """
        Creates a StateCollection from the states that have been
        added to the factory via `add_state`.

        Returns
        -------
        state_collection : StateCollection
        """
        state_collection = StateCollection()
        state_collection.data_frame = pandas.DataFrame(
                                        self.state_list,
                                        index=self.state_id_list)
        return state_collection


class StateCollection(object):
    """
    The states of a model. This data structures is used by
    AggregatedKineticModels to organize their states.

    Attributes
    ----------
    data_frame : pandas DataFrame
        Each row in the DataFrame corresponds to a state, and
        the columns are based on the attributes of the states.
    """
    def __init__(self):
        super(StateCollection, self).__init__()
        self.data_frame = None

    def __len__(self):
        return len(self.data_frame)

    def __str__(self):
        return self.data_frame.to_string()

    def iter_states(self):
        """
        Iterate through state collection.

        Returns
        -------
        state_id : string
        state_series : panda Series
            The attributes of a state as a Series.
        """
        for state_id, state_series in self.data_frame.iterrows():
            yield (state_id, state_series)

    def get_state_ids(self):
        """
        Returns
        -------
        s : StateIDCollection
        """
        s = StateIDCollection()
        s.state_id_list = self.data_frame.index.tolist()
        return s

    def sort(self, sort_column):
        """
        Parameters
        ----------
        sort_column : string
            Must correspond to an attribute of the states.
            Useful for organizing states according to aggregated class.
        """
        return self.data_frame.groupby(sort_column)

class StateIDCollection(object):
    """
    The identifier strings of the states of a model.

    Attributes
    ----------
    state_id_list : list
    """
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
        """
        Add another state id string to the collection.

        Parameters
        ----------
        state_id : string
        """
        self.state_id_list.append(state_id)

    def add_state_id_list(self, state_id_list):
        """
        Add a list of id strings to the collection.

        Parameters
        ----------
        state_id_list : list
        """
        self.state_id_list += state_id_list

    def from_state_collection(self, state_collection):
        """
        Replace current collection of id strings with those
        from the states in a StateCollection.

        Parameters
        ----------
        state_collection : StateCollection
        """
        self.state_id_list = state_collection.get_id_list()

    def as_list(self):
        return self.state_id_list
