import numpy
from palm.base.trajectory import TrajectorySegment, Trajectory

class DiscreteDwellSegment(TrajectorySegment):
    """
    Dwells consist of an aggregated class and a dwell duration.

    Parameters
    ----------
    segment_class : string
        The aggregated class observed during this segment of the trajectory.
    segment_duration : float
        The duration of this segment of the trajectory.
    """
    def __init__(self, segment_class, segment_duration):
        super(DiscreteDwellSegment, self).__init__()
        self.segment_class = segment_class
        self.segment_duration = segment_duration

    def __eq__(self, other_segment):
        condition1 = (self.segment_class == other_segment.segment_class)
        condition2 = (self.segment_duration == other_segment.segment_duration)
        return condition1 and condition2

    def get_class(self):
        return self.segment_class

    def get_duration(self):
        return self.segment_duration


class DiscreteStateTrajectory(Trajectory):
    """
    A sequence of trajectory segments. During each segment an observation
    is made, which corresponds to one of a finite number of discrete
    aggregated classes. Each segment lasts for a finite length of time.

    Attributes
    ----------
    segment_list : list
        The segments that make up the trajectory.
    cumulative_time_list : list
        The time elapsed since the start of the trajectory.
        Element `i` is the time elapsed up to the end of segment `i`.
    """
    def __init__(self):
        super(DiscreteStateTrajectory, self).__init__()
        self.segment_list = []
        self.cumulative_time_list = []

    def __len__(self):
        return len(self.segment_list)

    def __str__(self):
        full_str = ""
        for segment in self.segment_list:
            segment_class = segment.get_class()
            segment_duration = segment.get_duration()
            full_str += "%s,%.4e\n" % (segment_class, segment_duration)
        return full_str

    def __iter__(self):
        for segment in self.segment_list:
            yield segment

    def __eq__(self, other_trajectory):
        is_equal = True
        for my_segment, other_segment in zip(self, other_trajectory):
            if my_segment == other_segment:
                continue
            else:
                is_equal = False
                break
        return is_equal

    def add_segment(self, segment):
        """
        Add segment to trajectory. Assumes that this new segment comes after
        the previously added segments.

        Parameters
        ----------
        segment : TrajectorySegment
        """
        self.segment_list.append(segment)
        segment_duration = segment.get_duration()
        if len(self.cumulative_time_list) == 0:
            cumulative_time = 0.0 + segment_duration
        else:
            cumulative_time = self.cumulative_time_list[-1] + segment_duration
        self.cumulative_time_list.append(cumulative_time)

    def get_segment(self, segment_number):
        if segment_number < len(self.segment_list) and segment_number >= 0:
            return self.segment_list[segment_number]
        else:
            return None

    def get_cumulative_time(self, segment_number):
        if segment_number < len(self.segment_list):
            return self.cumulative_time_list[segment_number]
        else:
            return None

    def get_end_time(self):
        return self.cumulative_time_list[-1]

    def get_last_segment_number(self):
        return len(self) - 1

    def reverse_iter(self):
        reverse_range = range(len(self.segment_list))
        reverse_range.reverse()
        for i in reverse_range:
            yield (i, self.segment_list[i])

    def to_csv_str(self):
        csv_str = "class,dwell time\n"
        csv_str += str(self)
        return csv_str

    def as_continuous_traj_array(self):
        """
        Convert trajectory to a format that looks good for plotting.
        Output format:
        0.0 0
        1.0 0
        1.0 1
        1.1 1
        1.1 0
        3.6 0
        3.6 1
        3.8 1
        3.8 0

        Returns
        -------
        traj_array : numpy ndarray
        """
        class_to_signal_dict = {'dark':0, 'bright':1}
        time_list = []
        signal_list = []
        time_list.append(0.0)
        signal_list.append(class_to_signal_dict['dark'])
        for i, segment in enumerate(self):
            if i >= len(self)-1:
                break
            this_class = segment.get_class()
            this_signal = class_to_signal_dict[this_class]
            next_signal = class_to_signal_dict[self.get_segment(i+1).get_class()]
            this_cumulative_time = self.get_cumulative_time(i)
            time_list.append(this_cumulative_time)
            signal_list.append(this_signal)
            time_list.append(this_cumulative_time)
            signal_list.append(next_signal)
        last_segment = self.get_segment(len(self)-1)
        last_class = last_segment.get_class()
        last_signal = class_to_signal_dict[last_class]
        last_cumulative_time = self.get_cumulative_time(len(self)-1)
        time_list.append(last_cumulative_time)
        signal_list.append(last_signal)
        traj_array = numpy.array([time_list, signal_list]).T
        assert traj_array.shape[1] == 2
        return traj_array
