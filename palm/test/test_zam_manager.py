import nose.tools
from palm.zam_manager import ZamManager
import os, time

def remote_work_stub(*args, **kwargs):
    return 0.0

def remote_work_with_args_stub(*args, **kwargs):
    my_work = args[0] + args[1]
    return my_work

@nose.tools.istest
def test_zam_network():
    queue_name = 'test'
    task_manager = ZamManager('test', 1)
    task_manager.start(zam_network_runs_until_tasks_complete)
    time.sleep(10)
    # cmd = "scripts/zam_slave.py %s &" % queue_name
    # os.system(cmd)

def zam_network_runs_until_tasks_complete(task_manager):
    import nose.tools
    num_tasks = 5
    remote_work = remote_work_stub
    for i in xrange(num_tasks):
        print i
        task_manager.add_task(remote_work)
    unfinished_tasks = task_manager.count_unfinished_tasks()
    try:
        nose.tools.eq_(unfinished_tasks, self.num_tasks,
                       msg="%d tasks sent to task manager." % unfinished_tasks)
        task_manager.collect_results_from_completed_tasks()
        unfinished_tasks = task_manager.count_unfinished_tasks()
        nose.tools.eq_(unfinished_tasks, 0,
                       msg="%d tasks are still running." % unfinished_tasks)
    finally:
        print "Stopping task manager..."
        task_manager.stop()

    # @nose.tools.istest
    # def zam_network_accepts_tasks_with_arguments(self):
    #     self.setup()
    #     for i in xrange(self.num_tasks):
    #         arg1 = i
    #         arg2 = i+1
    #         self.task_manager.add_task(self.remote_work, arg1, arg2)
    #     unfinished_tasks = self.task_manager.count_unfinished_tasks()
    #     nose.tools.eq_(unfinished_tasks, self.num_tasks,
    #                    msg="%d tasks sent to task manager." % unfinished_tasks)
    #     results = self.task_manager.collect_results_from_completed_tasks()
    #     unfinished_tasks = self.task_manager.count_unfinished_tasks()
    #     nose.tools.eq_(unfinished_tasks, 0,
    #                    msg="%d tasks are still running." % unfinished_tasks)
    #     self.teardown()

def main():
    test_zam_network()

if __name__ == '__main__':
    main()