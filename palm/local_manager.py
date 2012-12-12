from base.task_manager import TaskManager

class LocalManager(TaskManager):
    """docstring for LocalManager"""
    def __init__(self):
        super(LocalManager, self).__init__()
        self.task_queue = []
    def start(self):
        pass
    def stop(self):
        pass
    def add_task(self, task, args):
        self.task_queue.append((task, args))
    def collect_results_from_completed_tasks(self):
        results = []
        for t in self.task_queue:
            task = t[0]
            args = t[1]
            result = task(args)
            results.append(result)
        self.task_queue = []
        return results
    def count_unfinished_tasks(self):
        return len(self.task_queue)
