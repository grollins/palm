#!/usr/bin/env python

"""
This module implements a remote slave that can recieve work from a master
and return the results. This module uses the U{http://twistedmatrix.com}
library to handle asynchronous network commuincation.
"""


import logging
import palm.util
logger = palm.util.SetupLogging(__name__)

from twisted.spread import pb
from twisted.internet import reactor, task, threads

import socket
import cPickle


class RemoteSlave(object):
    """
    Implements a remotely callable slave.

    This class implements a remotely callable slave that has the following features:

        1. It broadcasts its port information so that master nodes can find it.

        2. After a master has called 'register' on this object, then it will stop broadcasting.

        3. It has a ping method that should be called periodically by the server.

        4. It has a heartbeat timer. If we have not heard from the server in the last two heartbeats,
           then we shutdown.

        5. This class can execute arbitrary code sent by the master and return the results.


    To use this class::

        from zam.network.slave import RemoteSlave
        import logging

        # do any initialization here
        # e.g. setup logging
        logging.basicConfig(level=logging.INFO)

        # create an instance
        # the argument is an identifier for which server to contact
        # the server must be running with the same identifier
        slave = RemoteSlave('identifier')

        # start the slave
        slave.start()

        # this code will not execute until the slave has shutdown
        # do any cleanup here

    @ivar master_id: the identifier for our server
    @type master_id: string

    @ivar slave_server: the object that handles network connections
    @type slave_server: L{zam.network.slave.SlaveServer}

    @ivar listener: the object that listens for network connections

    @ivar port: the port we are listening on
    @type port: Integer

    @ivar heartbeat_time: Number of seconds between heartbeat checks
    @type heartbeat_time: Float

    @ivar heartbeat_task: A task that checks to see if the master has been pinging us
    @type heartbeat_task: L{twisted.internet.task.LoopingCall}

    @ivar broadcast_time: How long to wait between sending broadcast messages
    @type broadcast_time: Float

    @ivar broadcast_task: A task that sends a broadcast message with our port information.
    @type broadcast_task: L{twisted.internet.task.LoopingCall}
    """
    def __init__(self, master_id):
        """
        Initialize a RemoteSlave.

        @param master_id: the identifier for the master we are trying to contact
        @type  master_id: string
        """
        self.master_id = master_id
        self.slave_server = SlaveServer()
        self.listener = None
        self.port = None
        self.heartbeat_time = 1200
        self.heartbeat_task = None
        self.broadcast_time = 1
        self.broadcast_task = None
        self.socket = None


    def start(self):
        """
        Start listening for remote calls.

        Note: This call will start the twisted event loop. It will not return until the
        event loop has shutdown and the network services are no longer running.
        """
        # setup the listener and keep track of our port
        self.listener = reactor.listenTCP( 0, pb.PBServerFactory(self.slave_server) )
        self.port = self.listener.getHost().port
        logger.debug('I am on port %d, here me roar!' % self.port)

        # start the broadcast
        self.broadcast_task = task.LoopingCall(self.do_broadcast)
        self.broadcast_task.start(self.broadcast_time)

        #start the heartbeat
        self.heartbeat_task = task.LoopingCall(self.do_heartbeat)
        self.heartbeat_task.start(self.heartbeat_time)

        #start the event loop
        try:
            reactor.run()
        except:
            logger.debug( "RemoteSlave.start(): reactor already running" )


    def do_broadcast(self):
        """
        Send a broadcast message with our port number.
        """
        # multicast_addr = '224.0.0.1'
        multicast_addr = 'localhost'
        multicast_port = palm.util.PORT

        # if we're registered, we don't have to do anything
        if self.slave_server.registered:
            self.broadcast_task.stop()

        else:
            # Create the socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Make the socket multicast-aware, and set TTL.
            self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 20) # Change TTL (=20) to suit

            # Setup the data
            data = '%s\t%d' % (self.master_id, self.port)

            logger.debug('Multicasting port for %s.', self.master_id)
            # Send the data
            try:
                self.socket.sendto( data, (multicast_addr, multicast_port) )
            except:
                logger.info('Failed to send data to %s, %s' % (multicast_addr,multicast_port))


    def do_heartbeat(self):
        """
        Check to see if master has contacted us since last heartbeat. If not, then terminate.
        """
        if not self.slave_server.master_is_alive:
            logger.info('Have not heard from master in a long time. Terminating')
            reactor.callLater(0.5, reactor.stop)
        else:
            logger.debug('Checking heartbeat.')
            self.slave_server.master_is_alive = False



class SlaveServer(pb.Root):
    """
    Network server that responds to remote calls.

    @ivar registered: Has the server registered with us?
    @type registered: Boolean

    @ivar master_is_alive: Has the master pinged us recently?
    @type master_is_alive: Boolean

    @ivar busy: Are we busy?
    @type busy: Boolean
    """
    def __init__(self):
        self.registered = False
        self.master_is_alive = True
        self.busy = False


    def remote_terminate(self):
        """
        Terminate this SlaveServer.

        This is a remote method called by the master.
        """
        logger.info('Recieved termination signal from master. Terminating.')
        reactor.callLater(0.5, reactor.stop)


    def remote_do_work(self, command_string, data=None):
        """
        Execute task.

        This will unpack data into zam_sock_data and then exec command_string. It will take
        zam_sock_result as the output and return a pickled version of that.

        This method spawns a separate thread to execute the command in so that the rest
        of the network services can continue to run. It is an error to call this method
        a second time before the first task has finished.

        This is a remote method that is called by the server.
        """
        logger.debug('Recieved do_work command.')

        if self.busy:
            logger.error('Already busy.')
            raise RuntimeError('Recieved a do_work request when already busy.')
        else:
            self.busy = True

        # this function runs the commnad
        def evaluate_string(command_string, data):
            # unpickle the data
            unpickled_data = cPickle.loads(data)

            # setup an environemnt to execute command_string in
            exec_environment = {'zam_sock_data': unpickled_data, 'zam_sock_result': None}

            # run command string
            exec command_string in exec_environment

            # pickle and return the result
            result = exec_environment['zam_sock_result']
            result = cPickle.dumps(result)
            return result

        # sucess callback
        def cb(result):
            self.busy = False
            logger.debug('Finished worker thread.')
            logger.debug('Returning result to master.')
            return result

        # error callback
        def eb(failure):
            self.busy = False
            logger.error('There was an error executing do_work.')
            logger.error('The code was:')
            for line in command_string.splitlines():
                logger.error('    %s' % line)
            logger.error('The data was:')
            logger.error('    %s', cPickle.loads(data) )
            logger.error('The traceback was:')
            for line in failure.getTraceback().splitlines():
                logger.error('    %s', line)
            return failure

        logger.debug('Starting worker thread.')

        # run evaluate_string in a separate thread
        d = threads.deferToThread(evaluate_string, command_string, data)
        d.addCallback(cb)
        d.addErrback(eb)
        return d


    def remote_register(self):
        """
        Register server with this SlaveServer.

        Basically, this just stops us from sending more broadcast messages.

        This is a remote method called by the master.
        """
        logger.info('Recieved registration request.')
        self.registered = True
        self.master_is_alive = True


    def remote_ping(self):
        """
        This method is called by the master to let us know that it is still there.
        """
        logger.debug('Recieved ping from master.')
        self.master_is_alive = True
