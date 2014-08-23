import pandas
from copy import deepcopy
from .base.target_data import TargetData
from .discrete_state_trajectory import DiscreteStateTrajectory,\
                                           DiscreteDwellSegment


class BlinkTargetData(TargetData):
    """
    A dwell trajectory loaded from a file. The trajectory
    should be a series of dark and bright observations and
    the duration of each observation.
    Expecting csv file with this format:
        class,dwell time
        dark,1.5
        bright,0.3
        dark,1.2
        bright,0.1
        .
        .
        .

    Attributes
    ----------
    trajectory_factory : class
        A class that makes Trajectory objects.
    segment_factory : class
        A class that makes TrajectorySegment objects.
    trajectory : Trajectory
        Represents a time trace of dark and bright observations.
    filename : string
        The trajectory data is loaded from this path.
    """
    def __init__(self):
        super(BlinkTargetData, self).__init__()
        self.trajectory_factory = DiscreteStateTrajectory
        self.segment_factory = DiscreteDwellSegment
        self.trajectory = None
        self.filename = None

    def __len__(self):
        return len(self.trajectory)

    def load_data(self, data_file):
        """
        Load trajectory from file.

        Parameters
        ----------
        data_file : string
            Path of file to load.
        """
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
    An ensemble of trajectories.

    Attributes
    ----------
    trajectory_data_factory : class
        A class that makes TargetData objects.
    target_data_collection : list
        A list that holds the individual trajectories.
    paths : list
        Each trajectory is loaded from a file. `paths` is a list
        of the paths for these files.
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
        """
        Iterate over trajectories in collection.

        Returns
        -------
        trajectory : TargetData
            An individual trajectory from the collection.
        """
        for blink_target in self.target_data_collection:
            trajectory = blink_target.get_feature()
            yield trajectory

    def get_total_number_of_trajectory_segments(self):
        """
        Each trajectory is made up of segments (aka dwells).
        This method returns the total number of segments across
        all trajectories in the collection.
        """
        num_segments = 0
        for traj in self:
            num_segments += len(traj)
        return num_segments

    def get_feature_by_index(self, index):
        return self.target_data_collection[index]

    def load_data(self, data_file):
        """
        Load a trajectory and add it to the collection.

        Parameters
        ----------
        data_file : string
            Path of file to load.
        """
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
        """
        Make a new collection with a subset of the trajectories
        in this collection.

        Parameters
        ----------
        inds : list
            The indices of the trajectories to include in the subcollection.

        Returns
        -------
        my_clone : BlinkCollectionTargetData
            New collection which contains only the selected trajectories.
        """
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

    def get_bright_time_distribution(self):
        t_dist = []
        for traj in self:
            this_dist = traj.get_bright_time_distribution()
            t_dist += this_dist
        return pandas.Series(t_dist)

    def get_dark_time_distribution(self, exclude_first_dwell=True,
                                   exclude_last_dwell=True):
        t_dist = []
        for traj in self:
            excluded_dwells = []
            if exclude_first_dwell:
                excluded_dwells.append(0)
            if exclude_last_dwell:
                excluded_dwells.append(traj.get_last_segment_number())
            this_dist = traj.get_dark_time_distribution(excluded_dwells)
            t_dist += this_dist
        return pandas.Series(t_dist)

    def get_num_blink_distribution(self, exclude_first_dwell=True,
                                   exclude_last_dwell=True):
        num_blink_dist = []
        for traj in self:
            excluded_dwells = []
            if exclude_first_dwell:
                excluded_dwells.append(0)
            if exclude_last_dwell:
                excluded_dwells.append(traj.get_last_segment_number())
            this_num_blink = traj.get_num_blink(excluded_dwells)
            num_blink_dist.append(this_num_blink)
        return pandas.Series(num_blink_dist)

    def get_bleach_time_distribution(self):
        t_dist = []
        for traj in self:
            this_bleach_time = traj.get_bleach_time()
            t_dist.append(this_bleach_time)
        return pandas.Series(t_dist)            

    def get_activation_time_distribution(self):
        t_dist = []
        for traj in self:
            this_activation_time = traj.get_activation_time()
            t_dist.append(this_activation_time)
        return pandas.Series(t_dist)            

