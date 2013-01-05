import time
from base.task_manager import TaskManager
import network.master

class ZamManager(TaskManager):
    """
    A task manager that runs tasks across a network using
    a Twisted-based module previously implemented in Zam
    by JL MacCallum.
    """
    def __init__(self, queue_name, waiting_time=100):
        super(ZamManager, self).__init__()
        self.task_queue = None
        self.queue_name = queue_name
        self.max_iterations = 5
        self.task_list = []
        self.time_to_wait_before_checking_for_completed_tasks = waiting_time

    def start(self, main_fcn):
        self.task_queue = network.master.TaskQueue(self.queue_name)
        self.task_queue.set_main(main_fcn, self.task_queue)

    def stop(self):
        self.task_queue.stop()

    def add_task(self, task_fcn, *args,**kwargs):
        task = self.task_queue.add_task(task_fcn, *args,**kwargs)
        self.task_list.append(task)

    def collect_results_from_completed_tasks(self, noisy=False):
        results_list = []
        tasks_complete = 0

        for result in self.loop_over_results():
            results_list.append(result)

        return results_list

    def loop_over_results(self):
        iteration_count = 0
        while self.count_unfinished_tasks() > 0:
            completed_tasks = [task for task in self.task_list if task.done]
            print '%d tasks complete. %d remain.' % (len(completed_tasks),
                                                     self.count_unfinished_tasks())
            for task in completed_tasks:
                result = task.result()
                self.task_list.remove(task)
                yield result
            iteration_count += 1
            if iteration_count > self.max_iterations:
                print "Waiting too long for tasks to complete."
                break
            else:
                time.sleep(self.time_to_wait_before_checking_for_completed_tasks)

    def count_unfinished_tasks(self):
        return len(self.task_list)
