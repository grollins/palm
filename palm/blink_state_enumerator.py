from types import IntType
from .util import multichoose
from .state_collection import StateCollectionFactory


class SingleDarkState(object):
    """
    A macrostate for a BlinkModel with one dark microstate.
    The available microstates are `I`, `A`, `D`, and `B`.

    Attributes
    ----------
    initial_state_flag : bool
        This flag is used by BlinkModel when creating an initial
        probability vector. Expected to be true only for the
        macrostate in which `I` is the only microstate with nonzero
        population.

    Parameters
    ----------
    id_str : string
        A label that is used to identify to this macrostate.
    I,A,D,B : int
        The populations of the respective microstates.
    observation_class : string
        The aggregated class to which this macrostate belongs.

    """
    def __init__(self, id_str, I, A, D, B, observation_class):
        self.id = id_str
        self.I = I
        self.A = A
        self.D = D
        self.B = B
        self.observation_class = observation_class
        self.initial_state_flag = False
    def __str__(self):
        return "%s %s" % (self.id, self.observation_class)
    def as_array(self):
        return numpy.array([self.I, self.A, self.D, self.B])
    def get_id(self):
        return self.id
    def get_class(self):
        return self.observation_class
    def is_initial_state(self):
        return self.initial_state_flag
    def set_initial_state_flag(self):
        self.initial_state_flag = True
    def as_dict(self):
        return {'observation_class':self.get_class(),
                'I':self.I, 'A':self.A, 'D':self.D, 'B':self.B}


class DoubleDarkState(object):
    """
    A macrostate for a BlinkModel with one dark microstate.
    The available microstates are `I`, `A`, `D1`, `D2`, and `B`.

    Attributes
    ----------
    initial_state_flag : bool
        This flag is used by BlinkModel when creating an initial
        probability vector. Expected to be true only for the
        macrostate in which `I` is the only microstate with nonzero
        population.

    Parameters
    ----------
    id_str : string
        A label that is used to identify to this macrostate.
    I,A,D1,D2,B : int
        The populations of the respective microstates.
    observation_class : string
        The aggregated class to which this macrostate belongs.

    """
    def __init__(self, id_str, I, A, D1, D2, B, observation_class):
        self.id = id_str
        self.I = I
        self.A = A
        self.D1 = D1
        self.D2 = D2
        self.B = B
        self.observation_class = observation_class
        self.initial_state_flag = False
    def __str__(self):
        return "%s %s" % (self.id, self.observation_class)
    def as_array(self):
        return numpy.array([self.I, self.A, self.D1, self.D2, self.B])
    def get_id(self):
        return self.id
    def get_class(self):
        return self.observation_class
    def is_initial_state(self):
        return self.initial_state_flag
    def set_initial_state_flag(self):
        self.initial_state_flag = True
    def as_dict(self):
        return {'observation_class':self.get_class(),
                'I':self.I, 'A':self.A, 'D1':self.D1, 'D2':self.D2, 'B':self.B}


class SingleDarkStateEnumeratorFactory(object):
    """
    Creates a state enumerator for a BlinkModel with one dark state.

    Attributes
    ----------
    num_microstates : int

    Parameters
    ----------
    N : int
        The total number of fluorophores.
    state_factory : class
        Factory class for State objects.
    max_A : int
        Number of fluorophores that can be simultaneously active.
    """
    def __init__(self, N, state_factory=SingleDarkState, max_A=5):
        assert type(N) is IntType
        self.N = N
        self.state_factory = state_factory
        self.max_A = max_A
        self.num_microstates = len(['I', 'A', 'D', 'B'])

    def create_state_enumerator(self):
        """
        Creates a method that builds a StateCollection, made up of
        all possible macrostates in the model, subject to the
        constraint that no states with `A` > `max_A` are allowed.

        Returns
        -------
        enumerate_states : callable f()
            A method that builds a StateCollection.
        """
        def enumerate_states():
            """
            Builds a StateCollection for a model with one dark state.
            No states with `A` > `max_A` are allowed.

            Returns
            -------
            state_collection : StateCollection
                The allowed macrostates for the model.
            initial_state_id, final_state_id : string
                The identifier strings for the states where a time trace
                is expected to start and finish, respectively.
            """
            sc_factory = StateCollectionFactory()
            for this_count_list in multichoose(self.num_microstates, self.N):
                I = this_count_list[0]
                A = this_count_list[1]
                D = this_count_list[2]
                B = this_count_list[3]
                if A > self.max_A:
                    continue
                else:
                    if A > 0:
                        obs_class = 'bright'
                    else:
                        obs_class = 'dark'
                    id_str = "%d_%d_%d_%d" % (I, A, D, B)
                    this_state = self.state_factory(id_str, I, A, D, B,
                                                    obs_class)
                    if I == self.N:
                        initial_state_id = this_state.get_id()
                    elif B == self.N:
                        final_state_id = this_state.get_id()
                    else:
                        pass
                    sc_factory.add_state(this_state)
            state_collection = sc_factory.make_state_collection()
            return state_collection, initial_state_id, final_state_id
        return enumerate_states


class DoubleDarkStateEnumeratorFactory(object):
    """
    Creates a state enumerator for a BlinkModel with two dark states.

    Attributes
    ----------
    num_microstates : int

    Parameters
    ----------
    N : int
        The total number of fluorophores.
    state_factory : class
        Factory class for State objects.
    max_A : int
        Number of fluorophores that can be simultaneously active.
    """
    def __init__(self, N, state_factory=DoubleDarkState, max_A=5):
        assert type(N) is IntType
        self.N = N
        self.state_factory = state_factory
        self.max_A = max_A
        self.num_microstates = len(['I', 'A', 'D1', 'D2', 'B'])

    def create_state_enumerator(self):
        """
        Creates a method that builds a StateCollection, made up of
        all possible macrostates in the model, subject to the
        constraint that no states with `A` > `max_A` are allowed.

        Returns
        -------
        enumerate_states : callable f()
            A method that builds a StateCollection.
        """
        def enumerate_states():
            """
            Builds a StateCollection for a model with one dark state.
            No states with `A` > `max_A` are allowed.

            Returns
            -------
            state_collection : StateCollection
                The allowed macrostates for the model.
            initial_state_id, final_state_id : string
                The identifier strings for the states where a time trace
                is expected to start and finish, respectively.
            """
            sc_factory = StateCollectionFactory()
            for this_count_list in multichoose(self.num_microstates, self.N):
                I = this_count_list[0]
                A = this_count_list[1]
                D1 = this_count_list[2]
                D2 = this_count_list[3]
                B = this_count_list[4]
                if A > self.max_A:
                    continue
                else:
                    if A > 0:
                        obs_class = 'bright'
                    else:
                        obs_class = 'dark'
                    id_str = "%d_%d_%d_%d_%d" % (I, A, D1, D2, B)
                    this_state = self.state_factory(id_str, I, A, D1, D2, B,
                                                    obs_class)
                    if I == self.N:
                        initial_state_id = this_state.get_id()
                    elif B == self.N:
                        final_state_id = this_state.get_id()
                    else:
                        pass
                    sc_factory.add_state(this_state)
            state_collection = sc_factory.make_state_collection()
            return state_collection, initial_state_id, final_state_id
        return enumerate_states

