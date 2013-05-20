import abc

class TrajectorySegment(object):
    """
    Segment is an abstract class. Trajectories are divided into discrete
    segments. Each segment consists of an aggregated class and a duration.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_class(self):
        return

    @abc.abstractmethod
    def get_duration(self):
        return


class Trajectory(object):
    """
    Trajectory is an abstract class. A Trajectory is a series of segments.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_segment(self, segment):
        return

    @abc.abstractmethod
    def get_segment(self, segment_number):
        return

    @abc.abstractmethod
    def reverse_iter(self):
        return
