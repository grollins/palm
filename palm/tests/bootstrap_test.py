import nose.tools
from palm.blink_target_data import BlinkCollectionTargetData
from palm.bootstrap_selector import BootstrapSelector

@nose.tools.istest
def resampled_target_data_same_size_as_original_target_data():
    target_data = BlinkCollectionTargetData()
    target_data.load_data('./palm/tests/test_data/traj_directory.txt')
    bs_selector = BootstrapSelector()
    resampled_target_data = bs_selector.select_data(target_data)
    original_length = len(target_data)
    resampled_length = len(resampled_target_data)
    error_message = "Expected %d, got %d" % (original_length, resampled_length)
    nose.tools.eq_(original_length, resampled_length, error_message)

@nose.tools.istest
def all_elements_of_resampled_data_are_found_in_original_data():
    target_data = BlinkCollectionTargetData()
    target_data.load_data('./palm/tests/test_data/traj_directory.txt')
    bs_selector = BootstrapSelector()
    resampled_target_data = bs_selector.select_data(target_data)
    for data_element in resampled_target_data:
        error_message = "%s not found in original data" % str(data_element)
        nose.tools.ok_(target_data.has_element(data_element), error_message)

@nose.tools.istest
def resampled_data_is_not_in_the_same_order_as_original_data():
    target_data = BlinkCollectionTargetData()
    target_data.load_data('./palm/tests/test_data/traj_directory.txt')
    bs_selector = BootstrapSelector()
    resampled_target_data = bs_selector.select_data(target_data)
    combined_iterator = zip(target_data.iter_feature(),
                            resampled_target_data.iter_feature())
    mismatch_count = 0
    for original_feature, resampled_feature in combined_iterator:
        if original_feature == resampled_feature:
            continue
        else:
            mismatch_count += 1
    error_message = "Resampled data is identical to original data."
    nose.tools.ok_(mismatch_count > 0, error_message)
