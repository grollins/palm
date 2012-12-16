import abc

class TrajectorySegment(object):
    """Segment is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_class(self):
        return

    @abc.abstractmethod
    def get_duration(self):
        return


class Trajectory(object):
    """Trajectory is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_segment(self, segment):
        return

    @abc.abstractmethod
    def get_segment(self, segment_number):
        return
