import numpy
from palm.base.trajectory import TrajectorySegment, Trajectory

class DiscreteDwellSegment(TrajectorySegment):
    """Dwells consist of an observation class and a dwell duration."""
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
    Each segment of the trajectory corresponds to a
    discrete observation class (e.g. dark or bright).
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
        class_to_signal_dict = {'dark':0.0, 'bright':1.0}
        time_list = []
        signal_list = []
        # desired trajectory
        # 0.0 dark
        # 1.0 dark
        # 1.0 bright
        # 1.1 bright
        # 1.1 dark
        # 3.6 dark
        # 3.6 bright
        # 3.8 bright
        # 3.8 dark
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
