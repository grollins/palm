from types import IntType
from palm.util import multichoose
from palm.state_collection import StateCollectionFactory

class SingleDarkStateEnumeratorFactory(object):
    """
    Creates a state enumerator for a blink model with one dark state.
    """
    def __init__(self, N, state_factory, max_A):
        assert type(N) is IntType
        self.N = N
        self.state_factory = state_factory
        self.max_A = max_A
        self.num_microstates = len(['I', 'A', 'D', 'B'])

    def create_state_enumerator(self):
        def enumerate_states():
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
    Creates a state enumerator for a blink model with two dark states.
    """
    def __init__(self, N, state_factory, max_A):
        assert type(N) is IntType
        self.N = N
        self.state_factory = state_factory
        self.max_A = max_A
        self.num_microstates = len(['I', 'A', 'D1', 'D2', 'B'])

    def create_state_enumerator(self):
        def enumerate_states():
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
