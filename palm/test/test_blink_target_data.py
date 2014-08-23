import nose.tools
from ..blink_target_data import BlinkTargetData, BlinkCollectionTargetData


@nose.tools.istest
def loads_single_trajectory_with_correct_number_of_segments():
    ''' Expected format of file:
        class,dwell time
        dark,0.35
        bright,0.097
        dark,0.297
        bright,0.125
        dark,0.00519
    '''
    target_data = BlinkTargetData()
    data_file="./palm/test/test_data/short_blink_traj.csv"
    target_data.load_data(data_file)
    trajectory = target_data.get_feature()
    length_of_trajectory = len(trajectory)
    f = open(data_file, 'r')
    lines = f.readlines()
    expected_length = len(lines) - 1 # subtracting off header line
    error_msg = "Expected %d, got %d" % (expected_length, length_of_trajectory)
    nose.tools.eq_(length_of_trajectory, expected_length, error_msg)

@nose.tools.istest
def loads_many_trajectories_with_correct_number_of_trajectories():
    ''' 
    Expected format of directory file:
    /palm/tests/test_data/short_blink_traj.csv
    /palm/tests/test_data/short_blink_traj.csv
    /palm/tests/test_data/short_blink_traj.csv
    /palm/tests/test_data/short_blink_traj.csv

    Expected format of trajectory files:
        class,dwell time
        dark,0.35
        bright,0.097
        dark,0.297
        bright,0.125
        dark,0.00519
    '''
    target_data = BlinkCollectionTargetData()
    dir_file="./palm/test/test_data/traj_directory.txt"
    target_data.load_data(dir_file)
    trajectory_collection = target_data.get_feature()
    length_of_collection = len(trajectory_collection)
    f = open(dir_file, 'r')
    lines = f.readlines()
    expected_length = len(lines)
    error_msg = "Expected %d, got %d" % (expected_length, length_of_collection)
    nose.tools.eq_(length_of_collection, expected_length, error_msg)
