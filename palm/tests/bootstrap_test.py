import nose.tools
from palm.blink_target_data import BlinkCollectionTargetData
from palm.bootstrap_selector import BootstrapSelector

@nose.tools.istest
def resampled_target_data_has_expected_size():
    target_data = BlinkCollectionTargetData()
    target_data.load_data('./palm/tests/test_data/traj_directory.txt')
    bs_selector = BootstrapSelector()
    bs_size = 2
    resampled_target_data = bs_selector.select_data(target_data, size=bs_size)
    expected_length = bs_size
    resampled_length = len(resampled_target_data)
    error_message = "Expected %d, got %d" % (expected_length, resampled_length)
    nose.tools.eq_(expected_length, resampled_length, error_message)

@nose.tools.istest
def all_elements_of_resampled_data_are_elements_of_original_data():
    target_data = BlinkCollectionTargetData()
    target_data.load_data('./palm/tests/test_data/traj_directory.txt')
    bs_selector = BootstrapSelector()
    bs_size = 2
    resampled_target_data = bs_selector.select_data(target_data, size=bs_size)
    for data_element in resampled_target_data:
        error_message = "%s not found in original data" % str(data_element)
        nose.tools.ok_(target_data.has_element(data_element), error_message)

