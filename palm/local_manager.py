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
    def add_task(self, task, *args,**kwargs):
        self.task_queue.append((task, args, kwargs))
    def collect_results_from_completed_tasks(self, noisy=False):
        results = []
        for i, t in enumerate(self.task_queue):
            task = t[0]
            args = t[1]
            kwargs = t[2]
            result = task(*args, **kwargs)
            results.append(result)
            if noisy:
                print i
        self.task_queue = []
        return results
    def count_unfinished_tasks(self):
        return len(self.task_queue)
