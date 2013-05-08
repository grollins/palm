import pandas
from copy import deepcopy
from palm.base.target_data import TargetData
from palm.discrete_state_trajectory import DiscreteStateTrajectory,\
                                           DiscreteDwellSegment

class BlinkTargetData(TargetData):
    """
    One dwell trajectory loaded from a file. The trajectory
    should be a series of dark and bright observations and
    the duration of each observation.
    Expected format:
        class,dwell time
        dark,1.5
        bright,0.3
        dark,1.2
        bright,0.1
        .
        .
        .
    """
    def __init__(self):
        super(BlinkTargetData, self).__init__()
        self.trajectory_factory = DiscreteStateTrajectory
        self.segment_factory = DiscreteDwellSegment
        self.trajectory = None

    def __len__(self):
        return len(self.trajectory)

    def load_data(self, data_file):
        self.filename = data_file
        data_table = pandas.read_csv(data_file, header=0)
        self.trajectory = self.trajectory_factory()
        for segment_data in data_table.itertuples():
            segment_class = str(segment_data[1])
            segment_dwell_time = float(segment_data[2])
            new_segment = self.segment_factory(segment_class,
                                               segment_dwell_time)
            self.trajectory.add_segment(new_segment)

    def get_feature(self):
        return self.trajectory

    def get_target(self):
        return None

    def get_notes(self):
        return []

    def get_filename(self):
        return self.filename


class BlinkCollectionTargetData(TargetData):
    """
    An ensemble of trajectories. Each trajectory is expected
    to be in the format described for BlinkTargetData.
    """
    def __init__(self):
        super(BlinkCollectionTargetData, self).__init__()
        self.trajectory_data_factory = BlinkTargetData
        self.target_data_collection = None
        self.paths = None

    def __len__(self):
        return len(self.target_data_collection)

    def __iter__(self):
        for trajectory in self.iter_feature():
            yield trajectory

    def iter_feature(self):
        for blink_target in self.target_data_collection:
            trajectory = blink_target.get_feature()
            yield trajectory

    def get_total_number_of_trajectory_segments(self):
        num_segments = 0
        for traj in self:
            num_segments += len(traj)
        return num_segments

    def get_feature_by_index(self, index):
        return self.target_data_collection[index]

    def load_data(self, data_file):
        self.target_data_collection = []
        self.paths = []
        for traj_path in open(data_file, 'r'):
            traj_path = traj_path.strip()
            trajectory_data = self.trajectory_data_factory()
            trajectory_data.load_data(traj_path)
            self.target_data_collection.append(trajectory_data)
            self.paths.append(traj_path)

    def get_feature(self):
        return self.target_data_collection

    def get_target(self):
        return None

    def get_notes(self):
        return []

    def get_paths(self):
        return self.paths

    def make_copy_from_selection(self, inds):
        my_clone = deepcopy(self)
        new_data_collection = []
        new_paths = []
        for i in inds:
            this_traj = my_clone.target_data_collection[i]
            this_path = my_clone.paths[i]
            new_data_collection.append(this_traj)
            new_paths.append(this_path)
        my_clone.target_data_collection = new_data_collection
        my_clone.paths = new_paths
        return my_clone

    def has_element(self, trajectory_to_search_for):
        is_found = False
        for i, trajectory in enumerate(self):
            stop_condition = (trajectory == trajectory_to_search_for)
            if stop_condition:
               is_found = True
               break
            else:
                continue
        return is_found
