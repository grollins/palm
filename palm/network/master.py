#!/usr/bin/env python

"""
Implements a distributed task queue.

This class sets up the following:

    1. A multicast server that listens for remote slaves

    2. A ping routine that periodically sends ping requests
       to remote slaves to verify that they are still
       connected.

    3. A routine that periodically submits tasks to remote slaves.

This class is multithreaded. Everything except a user-supplied
main function executes in one thread using the Twisted event
driven network framework, while the main function executes
on a separate thread.

There are six main functions that must be called. First, we must
create new instance of the task queue. The argument is a unique
id so that we can have multiple task queues on the network at
the same time. Next, we need to set our user defined main function.
This will be called with the specified arguments by the task queue.
The task queue itself should be an argument so that we can make
calls within our main function. Next, we should start the task
queue. Within our main function, we should make calls to add_task
to submit tasks for remote execution. This returns a Task object
that has a result() method that will block until the result of
the remote calculation is available. Finally, we shoud call stop
to shutdown the task queue.::
    tq = TaskQueue('unique_id')

    def foo(tq, param):
        task = tq.add_task(....)
        result = task.result()
        tq.stop()

    tq.set_main(foo, tq, 10)
    tq.start()
"""



import logging
import palm.util
logger = palm.util.SetupLogging(__name__)

from twisted.spread import pb
from twisted.internet import reactor, defer
from twisted.internet.task import LoopingCall
from twisted.internet.protocol import DatagramProtocol

import inspect
import threading
import collections
import cPickle


class BroadcastMonitor(DatagramProtocol):
    """
    This class handles recieving broadcast messages.

    This class listens on palm.util.PORT for messages from remote slaves.
    The slaves will broadcast the port that they are listening on and
    the identifier of the master they are looking for. If our id matches,
    We log it and then call L{TaskQueue.add_client}.

    @ivar task_queue: the task queue that we are listening on behalf of
    @type task_queue: L{TaskQueue}

    @ivar master_id: the unique identifier we are listening for
    @type master_id: String
    """
    def __init__(self, task_queue, master_id):
        """
        Initialize

        @param task_queue: task queue to listen on behalf of
        @type  task_queue: L{TaskQueue}

        @param master_id: unique id we are listening for
        @type  master_id: String
        """
        self.task_queue = task_queue
        self.master_id = master_id


    def startProtocol(self):
        """
        Called by Twisted when we start listening.
        """
        logger.info('Broadcast monitor started listening.')
        self.transport.joinGroup('224.0.0.1')


    def datagramReceived(self, datagram, address):
        """
        Called by Twisted when we recieve a UDP packet.

        @param datagram: the datagram recieved

        @param address: the address of the remote host
        @type  address: tuple: (address, port)
        """
        master_id, remote_port = datagram.split()

        # Only respond if this multicast was for us
        if master_id == self.master_id:
            remote_address = address[0]
            remote_port = int(remote_port)
            logger.info( 'Reveived multicast from %s:%d' % (remote_address, remote_port) )
            self.task_queue.add_slave(remote_address, remote_port)
        else:
            logger.debug('Received multicast for %s; my id is %s; ignoring.' % (master_id, self.master_id) )



class Task(object):
    """
    This class implements a task that is to be distributed.

    In general, tasks will be created and returned to you by
    L{TaskQueue.add_task}.

    @ivar id: unique identifier
    @type id: Integer

    @ivar function: the function to be called.
    @type function: function

    @ivar pos_args: the positional arguments for function
    @type pos_args: list

    @ivar kw_args: the keyword arguments for function
    @type kw_args: dict

    @ivar done: are we done yet?
    @type done: Boolean

    @ivar event: this event is triggered when we the task is finished
    @type event: L{threading.Event}
    """
    def __init__(self, id_, function, pos_args, kw_args):
        """
        Initialize a new Task.

        @param id_: unique identifier
        @type id_: Integer

        @param function: the function to be called.
        @type function: function

        @param pos_args: the positional arguments for function
        @type pos_args: list

        @param kw_args: the keyword arguments for function
        @type kw_args: dict
        """
        self.id = id_
        self.function = function
        self.function_source = inspect.getsource(self.function)
        self.pos_args = pos_args
        self.kw_args = kw_args
        self.done = False
        self.event = threading.Event()
        self._result = None

        self.data = cPickle.dumps( [self.pos_args, self.kw_args], -1 )

        # setup the Deferred
        self.deferred = defer.Deferred()
        self.deferred.addCallback(self._receive_result_callback)


    def get_remote_command(self):
        """
        Get string representation of remote command and data.
        """
        # setup the command string
        command_string = 'pos_args = zam_sock_data[0]\n'
        command_string += 'kw_args = zam_sock_data[1]\n'
        command_string += self.function_source.strip() + '\n'
        command_string += 'zam_sock_result = %s(*pos_args, **kw_args)' % self.function.__name__

        # pickle the data
        # data = [self.pos_args, self.kw_args]
        # data = pickle.dumps(data, -1)
        data = self.data

        return command_string, data


    def _receive_result_callback(self, result):
        """
        This function is called when the callbacks on self.deferred are triggered.

        When the remote calculation is completed, this function is called and we update
        self._results and set self.event.
        """
        self._result = cPickle.loads(result)
        self.done = True
        self.event.set()
        return self._result


    def result(self):
        """
        Get the result of a remote calculation.

        This function blocks until the result is available.
        """
        self.event.wait()
        return self._result



class TaskQueue(object):
    """
    Manages a queue of remote calculations.
    """
    def __init__(self, master_id):
        """
        Initialize

        @param master_id: unique identifier for this task queue
        @type  master_id: String
        """
        logging.info('Starting TaskQueue.')
        self.master_id = master_id

        self._next_task_id = 0
        self._next_slave_id = 0

        # dicts to hold slaves
        self.slaves = dict()
        self.release_idle_slaves = False

        # lists of tasks
        # collections.deque is thread-safe
        self.queued_tasks = collections.deque()

        # the user function that will be run in a separate thread
        self.main_loop = None
        self.main_loop_pos_args = []
        self.main_loop_kw_args = dict()

        # main loop thread
        # this is started later
        self.main_loop_thread = None

        # setup the work submission task
        self.submission_time = 1
        self.work_submission_task = LoopingCall(self._submit_work)
        self.work_submission_task.start(self.submission_time)

        # setup the slave ping routines
        self.slave_ping_time = 300
        self.ping_task = LoopingCall(self._ping_slaves)
        self.ping_task.start(self.slave_ping_time)


    def set_main(self, func, *pos_args, **kw_args):
        """
        Set the main loop function.

        This function sets the user supplied function that will be
        run on the task queue. This function will be called with the
        supplied arguments. One of the arguments should be the this
        task queue itself, so that the user supplied function can
        call add_task on us.

        Example::
            def do_foo(tq, x, y):
                tq.add_task(..blah..)

            tq = TaskQueue('test')
            tq.set_main(do_foo, tq, y=10, x=2)
            tq.start()

        @param func: function to call
        @type  func: callable
        """
        self.main_loop = func
        self.main_loop_pos_args = pos_args
        self.main_loop_kw_args = kw_args


    def start(self):
        """
        Start the task queue.

        This starts the broadcast listener, the ping loop, and the work
        submission loop. The user-supplied main_loop starts after five
        seconds.
        """
        reactor.callLater(1, self._start_main_loop_thread)
        # reactor.callLater(600, reactor.stop)
        reactor.listenMulticast( palm.util.PORT, BroadcastMonitor(self, self.master_id), listenMultiple=True )
        try:
            reactor.run()
        except:
            logger.debug( "TaskQueue.start(): reactor already running" )


    def stop(self):
        """
        Shutdown the task queue.

        This calls terminate on all slaves, waits for the user-supplied main_loop to
        finish, and then shuts down Twisted.
        """
        logger.info('Shutting down master.')
        d = defer.DeferredList( [slave.remote.callRemote('terminate') for slave in self.slaves.values()],
            consumeErrors=True)
        d.addCallback(self._stop_main_loop_thread)


    def add_task(self, func, *pos_args, **kw_args):
        """
        Add a remote task.

        @param func: function to execute remotely
        @type func: function

        @param pos_args: positional arguments to func
        @param kw_args: keyword arguments to function.

        @returns: a L{Task} instance representing this calculation
        @rtype: L{Task}

        Example::
            tq = TaskQueue()
            task = tq.add_task(some_func, 1, 2, foo=17)
            result = task.result()
        """
        task = Task(self._next_task_id, func, pos_args, kw_args)
        self._next_task_id += 1
        self.queued_tasks.append(task)
        logger.info('Added task %d to TaskQueue.', task.id)

        return task


    def cancel_queued_tasks(self):
        '''
        Cancel all queued tasks.
        '''
        self.queued_tasks.clear()


    # ============================
    # = Private member functions =
    # ============================
    def _start_main_loop_thread(self):
        """
        Start the main loop thread.
        """
        self.main_loop_thread = threading.Thread(target=self.main_loop, args=self.main_loop_pos_args, kwargs=self.main_loop_kw_args)
        self.main_loop_thread.start()


    def _stop_main_loop_thread(self, __):
        """
        Wait for the main_loop_thread to finish and kill the reactor.
        """
        try:
            self.main_loop_thread.join()
        finally:
            reactor.callLater(5, reactor.stop)


    def _submit_work(self):
        """
        Submit work to slaves.

        Called autmatically by Twisted.
        """
        logger.debug('Called submit work.')

        count = 0

        # get all of the idle slaves
        idle_slaves = [slave for slave in self.slaves.values() if not slave.busy]

        # check to see if we are supposed to keep submitting work
        if self.release_idle_slaves and self.queued_tasks:
            logger.info('Not releasing idle slaves because there are still queued tasks.')
        elif self.release_idle_slaves:
            # call terminate on all idle slaves
            for slave in idle_slaves:
                try:
                    logger.debug('Dropping slave %s:%d', slave.address, slave.port)
                    d = slave.remote.callRemote('terminate')
                except pb.DeadReferenceError:
                    # we expect to get this exception because the slave has now terminated
                    # so now we remove the now-dead slave from the list
                    del( self.slaves[slave.id] )
            # we just return now because we aren't supposed to be submitting new work
            return

        for slave in idle_slaves:
            if self.queued_tasks:
                # get the task
                task = self.queued_tasks.popleft()
                logger.info('Starting task %d on slave %s:%d', task.id, slave.address, slave.port)

                # update task info
                task.running = True

                slave.busy = True
                slave.task = task

                command_string, data = task.get_remote_command()
                d = slave.remote.callRemote('do_work', command_string, data)
                d.addCallback(self._slave_work_done_callback, slave)
                d.addErrback(self._slave_work_errback, slave, task)
                slave.deferred = d
                count = count + 1
                if count > 10:
                    break
            else:
                logger.debug('Task queue is empty.')
                break

    def _slave_work_done_callback(self, result, slave):
        """
        Called when a slave completes a remote call.
        """
        logger.info('Got result for task %d from %s:%d', slave.task.id, slave.address, slave.port)

        # unmark the slave as done
        slave.busy = False
        task = slave.task
        slave.task = None
        slave.deferred = None

        # pass result along to task
        task.deferred.callback(result)


    def _slave_work_errback(self, failure, slave, task):
        """
        Called when there was an error during remote call.
        """
        logger.info('Task %d on slave %s:%d failed. Resubmitting task.', task.id, slave.address, slave.port)

        print failure
        # TODO: do something smart depending on the type of failure
        # we should requeue tasks that died due to network issues
        # but, we should kill the whole master if a remote dies
        # because of an unhandled exception.

        # remove the dead slave from our list of slaves
        self._remove_dead_slave(slave)

        # resubmit the task at the front of the queue
        task.running = False
        self.queued_tasks.appendleft(task)


    def add_slave(self, address, port):
        """
        Add a new slave at specified address and port.
        """
        # TODO: needs errBack

        # first check to see if we're already talking to this guy
        # if we are, we can just return because we already know about
        # him
        # for s in self.slaves.values():
        #     if s.address == address and s.port == port:
        #         return
        #
        factory = pb.PBClientFactory()
        reactor.connectTCP(address, port, factory)
        d = factory.getRootObject()

        def eb(f, address, port):
            logger.info('Something bad happened at %s:%s' % (address, port) )

        d.addCallback(self._new_slave_callback, address, port)
        d.addErrback(eb, address, port)


    def _new_slave_callback(self, remote, address, port):
        """
        Called when a new slave joins the queue.
        """
        for s in self.slaves.values():
            if (address == s.address) and (port == s.port):
                logger.info('Slave %s:%d was already present. Skipping.', address, port)
                return

        slave = self._new_slave(address, port)
        slave.remote = remote
        slave.remote.callRemote('register')
        logger.info('Added new slave %d at %s:%d', slave.id, address, port)

    def _new_slave(self, address, port):
        """
        Called to create a new slave at specified address and port.
        """
        slave = Slave(self._next_slave_id, address, port)
        self._next_slave_id += 1
        assert not slave.id in self.slaves
        self.slaves[slave.id] = slave
        return slave


    def _remove_slave(self, slave_id):
        """
        Remove a slave from our list of slaves.

        If the slave was busy, then requeue to work.
        """
        assert slave_id in self.slaves
        slave = self.slaves[slave_id]

        # if the slave isn't busy we can just delete it
        if not slave.busy:
            del( self.slaves[slave_id] )
            logger.info('Removed slave %s:%d', slave.address, slave.port)

        # TODO: should we actually be doing this?
        # the _slave_work_errback alread does this

        # the slave was busy, so we need to deal with whatever it was running
        else:
            task = slave.task
            d = slave.deferred
            del( self.slaves[slave_id] )
            logger.info('Removed busy slave %s:%d', slave.address, slave.port)

            logger.info('Resubmitting task %d.' % task.id)
            task.running = False
            self.queued_tasks.append(task)


    def _remove_dead_slave(self, slave):
        """
        Remove a dead slave from our list of slaves.
        """
        logger.info('Slave %s:%d not responding. Dropped.', slave.address, slave.port)
        if slave.id in self.slaves:
            del( self.slaves[slave.id] )
        slave.deferred = None
        slave.task = None


    def _ping_slaves(self):
        """
        Send a ping command to all of our slaves.
        """
        for slave in self.slaves.values():
            def eb(f, the_slave):
                self._remove_dead_slave(the_slave)
            def cb(r, the_slave):
                logger.info('Ping successful with %s:%d' % (the_slave.address, the_slave.port) )
            try:
                logger.info('Pinging slave %s:%d' % (slave.address, slave.port) )
                d = slave.remote.callRemote('ping')
                d.addErrback(eb, slave)
                d.addCallback(cb, slave)
            except:
                self._remove_dead_slave(slave)



class Slave(object):
    """
    Data structure to represent a remote slave on the server side.

    @ivar id: unique id for slavw
    @type id: Integer

    @ivar address: ip address
    @type address: String

    @ivar port: remote port
    @type port: Integer

    @ivar task: current task running on remote slave
    @type task: L{Task}

    @ivar deferred: deferred that will trigger when remote calculation finishes
    @type deferred: L{twisted.internet.defer.Deferred}

    @ivar remote: remote object to run calls on
    @type remote: L{twisted.spread.pb.Root}

    @ivar busy: is this slave busy?
    @type busy: Boolean
    """
    def __init__(self, id_, address, port):
        self.id = id_
        self.address = address
        self.port = port
        self.task = None
        self.deferred = None
        self.remote = None
        self.busy = False