import os
import time
from palm.base.task_manager import TaskManager
from IPython.parallel import Client

class IPythonManager(TaskManager):
    """
    A task manager that runs tasks in the IPython parallel environment.
    """
    def __init__(self, refresh_time):
        super(IPythonManager, self).__init__()
        self.client = None
        self.task_queue = None
        self.task_list = []
        self.refresh_time = refresh_time # seconds
        self.max_iterations = 100

    def start(self):
        # os.system("ipcluster start -n 2 &")
        self.client = Client()
        self.task_queue = self.client.load_balanced_view()

    def stop(self):
        # os.system("ipcluster stop")
        return

    def add_task(self, task, *args, **kwargs):
        async_task = self.task_queue.apply_async(task, *args, **kwargs)
        self.task_list.append( async_task )

    def collect_results_from_completed_tasks(self, noisy=False):
        tasks_complete = 0
        results_list = []
        for result in self.loop_over_results():
            results_list.append(result)
            if noisy:
                tasks_complete += 1
                tasks_remaining = self.count_unfinished_tasks()
                print '%d tasks complete, %d remain' % (tasks_complete,
                                                        tasks_remaining)
        return results_list

    def loop_over_results(self):
        iteration_count = 0
        while self.count_unfinished_tasks() > 0:
            completed_task_list = []
            for task in self.task_list:
                if task.ready():
                    completed_task_list.append(task)
            for completed_task in completed_task_list:
                this_result = completed_task.get()
                self.task_list.remove(completed_task)
                yield this_result
            iteration_count += 1
            if iteration_count > self.max_iterations:
                print "Waiting too long for tasks to complete."
                break
            else:
                time.sleep(self.refresh_time)

    def count_unfinished_tasks(self):
        return len(self.task_list)
