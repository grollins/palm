import nose.tools
from palm.local_manager import LocalManager

def remote_work_stub(*args, **kwargs):
    return 0.0

def remote_work_with_args_stub(*args, **kwargs):
    my_work = args[0] + args[1]
    return my_work

@nose.tools.istest
class LocalNetworkTest(object):
    def setup(self):
        self.task_manager = LocalManager()
        self.task_manager.start()
        self.num_tasks = 5
        self.remote_work = remote_work_stub
        self.time_to_wait_for_tasks = 3 # seconds

    def teardown(self):
        self.task_manager.stop()

    @nose.tools.istest
    def local_network_runs_until_tasks_complete(self):
        self.setup()
        args = ()
        for i in xrange(self.num_tasks):
            self.task_manager.add_task(self.remote_work)
        unfinished_tasks = self.task_manager.count_unfinished_tasks()
        nose.tools.eq_(unfinished_tasks, self.num_tasks,
                       msg="%d tasks sent to task manager." % unfinished_tasks)
        self.task_manager.collect_results_from_completed_tasks()
        unfinished_tasks = self.task_manager.count_unfinished_tasks()
        nose.tools.eq_(unfinished_tasks, 0,
                       msg="%d tasks are still running." % unfinished_tasks)
        self.teardown()

    @nose.tools.istest
    def local_network_accepts_tasks_with_arguments(self):
        self.setup()
        for i in xrange(self.num_tasks):
            arg1 = i
            arg2 = i+1
            args = (arg1, arg2)
            self.task_manager.add_task(self.remote_work, arg1, arg2)
        unfinished_tasks = self.task_manager.count_unfinished_tasks()
        nose.tools.eq_(unfinished_tasks, self.num_tasks,
                       msg="%d tasks sent to task manager." % unfinished_tasks)
        results = self.task_manager.collect_results_from_completed_tasks()
        unfinished_tasks = self.task_manager.count_unfinished_tasks()
        nose.tools.eq_(unfinished_tasks, 0,
                       msg="%d tasks are still running." % unfinished_tasks)
        self.teardown()
