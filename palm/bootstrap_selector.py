from sklearn.cross_validation import Bootstrap
from .base.data_selector import DataSelector


class BootstrapSelector(DataSelector):
    """Randomly samples a data set with replacement."""
    def __init__(self):
        super(BootstrapSelector, self).__init__()

    def select_data(self, target_data, size):
        n = len(target_data)
        assert size <= n-1
        bs = Bootstrap(n, 1, train_size=n-1)
        train_index, test_index = bs.__iter__().next()
        train_index = list(train_index)
        inds = train_index[:size]
        new_target_data = target_data.make_copy_from_selection(inds)
        return new_target_data
