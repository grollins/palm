from util import multichoose
from types import IntType

class SingleDarkStateEnumeratorFactory(object):
    def __init__(self, N, state_factory, max_A):
        assert type(N) is IntType
        self.N = N
        self.state_factory = state_factory
        self.max_A = max_A

    def create_state_enumerator(self):
        def enumerate_states():
            # There are 4 states (I, A, D, B)
            state_list = []
            i = 0
            for this_count_list in multichoose(4, self.N):
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
                        this_state.set_initial_state_flag()
                    state_list.append(this_state)
                    i += 1
            return state_list
        return enumerate_states

class DoubleDarkStateEnumeratorFactory(object):
    def __init__(self, N, state_factory, max_A):
        assert type(N) is IntType
        self.N = N
        self.state_factory = state_factory
        self.max_A = max_A

    def create_state_enumerator(self):
        def enumerate_states():
            # There are 4 states (I, A, D1, D2, B)
            state_list = []
            i = 0
            for this_count_list in multichoose(5, self.N):
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
                        this_state.set_initial_state_flag()
                    state_list.append(this_state)
                    i += 1
            return state_list
        return enumerate_states
