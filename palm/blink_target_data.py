import pandas
import base.target_data
from discrete_state_trajectory import DiscreteStateTrajectory,\
                                      DiscreteDwellSegment
from copy import deepcopy

class BlinkTargetData(base.target_data.TargetData):
    """ Expected format
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

    def load_data(self, data_file):
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

class BlinkCollectionTargetData(base.target_data.TargetData):
    """docstring for BlinkCollectionTargetData"""
    def __init__(self):
        super(BlinkCollectionTargetData, self).__init__()
        self.trajectory_data_factory = BlinkTargetData
        self.target_data_collection = None

    def __len__(self):
        return len(self.target_data_collection)

    def __iter__(self):
        for trajectory in self.iter_feature():
            yield trajectory

    def iter_feature(self):
        for blink_target in self.target_data_collection:
            trajectory = blink_target.get_feature()
            yield trajectory

    def load_data(self, data_file):
        self.target_data_collection = []
        for traj_path in open(data_file, 'r'):
            traj_path = traj_path.strip()
            trajectory_data = self.trajectory_data_factory()
            trajectory_data.load_data(traj_path)
            self.target_data_collection.append(trajectory_data)

    def get_feature(self):
        return self.target_data_collection

    def get_target(self):
        return None

    def get_notes(self):
        return []

    def make_copy_from_selection(self, inds):
        my_clone = deepcopy(self)
        new_data_collection = []
        for i in inds:
            this_traj = my_clone.target_data_collection[i]
            new_data_collection.append(this_traj)
        my_clone.target_data_collection = new_data_collection
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
