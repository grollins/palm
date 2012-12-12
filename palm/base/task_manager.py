import abc

class TaskManager(object):
    """TaskManager is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def start(self):
        return

    @abc.abstractmethod
    def stop(self):
        return

    @abc.abstractmethod
    def add_task(self, task, args):
        return

    @abc.abstractmethod
    def collect_results_from_completed_tasks(self):
        return

    @abc.abstractmethod
    def count_unfinished_tasks(self):
        return

