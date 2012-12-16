import base.trajectory

class DiscreteDwellSegment(base.trajectory.TrajectorySegment):
    """docstring for DiscreteDwellSegment"""
    def __init__(self, segment_class, segment_duration):
        super(DiscreteDwellSegment, self).__init__()
        self.segment_class = segment_class
        self.segment_duration = segment_duration

    def get_class(self):
        return self.segment_class

    def get_duration(self):
        return self.segment_duration


class DiscreteStateTrajectory(base.trajectory.Trajectory):
    """docstring for DiscreteStateTrajectory"""
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
            full_str += "%s  %.4e\n" % (segment_class, segment_duration)
        return full_str

    def __iter__(self):
        for segment in self.segment_list:
            yield segment

    def add_segment(self, segment):
        self.segment_list.append(segment)
        segment_duration = segment.get_duration()
        if len(self.cumulative_time_list) == 0:
            cumulative_time = 0.0 + segment_duration
        else:
            cumulative_time = self.cumulative_time_list[-1] + segment_duration
        self.cumulative_time_list.append(cumulative_time)

    def get_segment(self, segment_number):
        if segment_number < len(self.segment_list):
            return self.segment_list[segment_number]
        else:
            return None

    def get_cumulative_time(self, segment_number):
        if segment_number < len(self.segment_list):
            return self.cumulative_time_list[segment_number]
        else:
            return None

