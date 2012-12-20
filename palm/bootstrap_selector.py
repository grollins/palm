import base.data_selector
from sklearn.cross_validation import Bootstrap

class BootstrapSelector(base.data_selector.DataSelector):
    """docstring for BootstrapSelector"""
    def __init__(self):
        super(BootstrapSelector, self).__init__()

    def select_data(self, target_data):
        n = len(target_data)
        bs = Bootstrap(n, 1, train_size=n-1, test_size=1)
        train_index, test_index = bs.__iter__().next()
        train_index = list(train_index)
        test_index = list(test_index)
        inds = train_index + test_index
        new_target_data = target_data.make_copy_from_selection(inds)
        return new_target_data
