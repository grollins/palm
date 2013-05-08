import numpy
import pandas

from palm.base.model import Model
from palm.base.model_factory import ModelFactory
from palm.base.parameter_set import ParameterSet
from palm.base.target_data import TargetData
from palm.aggregated_kinetic_model import AggregatedKineticModel
from palm.discrete_state_trajectory import DiscreteStateTrajectory,\
                                           DiscreteDwellSegment


class State(object):
    def __init__(self, id_str, observation_class):
        self.id = id_str
        self.observation_class = observation_class
        self.initial_state_flag = False
    def __str__(self):
        return "%s %s" % (self.id, self.observation_class)
    def as_array(self):
        return None
    def get_id(self):
        return self.id
    def get_class(self):
        return self.observation_class
    def is_initial_state(self):
        return self.initial_state_flag
    def set_initial_state_flag(self):
        self.initial_state_flag = True
    def as_dict(self):
        return {'observation_class':self.get_class()}

class Route(object):
    '''
    A generic route class for aggregated kinetic models.
    '''
    def __init__(self, id_str, start_state_id, end_state_id, rate_id,
                 multiplicity):
        self.id = id_str
        self.start_state_id = start_state_id
        self.end_state_id = end_state_id
        self.rate_id = rate_id
        self.multiplicity = multiplicity

    def __str__(self):
        my_str = "%s %s %s %s %d" % (
                    self.id, self.start_state_id, self.end_state_id,
                    self.rate_id, self.multiplicity)
        return my_str

    def get_id(self):
        return self.id
    def get_start_state(self):
        return self.start_state_id
    def get_end_state(self):
        return self.end_state_id
    def get_multiplicity(self):
        return self.multiplicity
    def as_dict(self):
        return {'start_state':self.start_state_id,
                'end_state':self.end_state_id,
                'rate_id':self.rate_id,
                'multiplicity':self.multiplicity}

class SimpleParameterSet(ParameterSet):
    """
    Parameters for a simple two-state model.
    """
    def __init__(self):
        super(SimpleParameterSet, self).__init__()
        self.parameter_dict = {'log_k1':-1.0, 'log_k2':-1.0}
        self.bounds_dict = {'log_k1':(None, None),
                            'log_k2':(None, None)}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, parameter_name, parameter_value):
        self.parameter_dict[parameter_name] = parameter_value

    def get_parameter(self, parameter_name):
        return self.parameter_dict[parameter_name]

    def as_array(self):
        log_k1 = self.get_parameter('log_k1')
        log_k2 = self.get_parameter('log_k2')
        return numpy.array([log_k1, log_k2])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           log_k1, log_k2
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_k1', parameter_array[0])
        self.set_parameter('log_k2', parameter_array[1])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        log_k1_bounds = self.bounds_dict['log_k1']
        log_k2_bounds = self.bounds_dict['log_k2']
        bounds = [log_k1_bounds, log_k2_bounds]
        return bounds


class SimpleModelFactory(ModelFactory):
    """
    This factory class creates a simple two-state aggregated
    kinetic model.
    """
    def __init__(self):
        super(SimpleModelFactory, self).__init__()
        self.state_factory = State
        self.route_factory = Route

    def create_model(self, parameter_set):
        self.parameter_set = parameter_set
        state_enumerator = self.state_enumerator_factory()
        route_mapper = self.route_mapper_factory()
        new_model = SimpleModel(state_enumerator, route_mapper,
                                self.parameter_set)
        return new_model

    def state_enumerator_factory(self):
        def enumerate_states():
            state_list = []
            A = self.state_factory('A', 'green')
            A.set_initial_state_flag()
            B = self.state_factory('B', 'orange')
            state_list.append(A)
            state_list.append(B)
            return state_list
        return enumerate_states

    def log_k1_factory(self):
        log_k1 = self.parameter_set.get_parameter('log_k1')
        def log_k1_fcn(t):
            return log_k1
        return log_k1_fcn

    def log_k2_factory(self):
        log_k2 = self.parameter_set.get_parameter('log_k2')
        def log_k2_fcn(t):
            return log_k2
        return log_k2_fcn

    def route_mapper_factory(self):
        def map_routes(states, state_index_dict):
            log_k1_fcn = self.log_k1_factory()
            log_k2_fcn = self.log_k2_factory()
            A_to_B = self.route_factory('A', 'B', log_k1_fcn)
            B_to_A = self.route_factory('B', 'A', log_k2_fcn)
            route_list = []
            route_list.append(A_to_B)
            route_list.append(B_to_A)
            return route_list
        return map_routes


class SimpleModel(AggregatedKineticModel):
    """
    A simple two-state model with green and orange observation classes.
    """
    def __init__(self, state_enumerator, route_mapper, parameter_set):
        super(SimpleModel, self).__init__(state_enumerator, route_mapper,
                                          parameter_set)

    def get_initial_population_array(self):
        initial_population_array = numpy.zeros([1, self.get_num_states('green')])
        initial_population_array[0,0] = 1.0
        return initial_population_array

class SimpleTargetData(TargetData):
    """
    One dwell trajectory loaded from a file. The trajectory
    should be a series of green and orange observations and
    the duration of each observation.
    
    Expected format:
        class,dwell time
        green,1.5
        orange,0.3
        green,1.2
        orange,0.1
        .
        .
        .
    """
    def __init__(self):
        super(SimpleTargetData, self).__init__()
        self.trajectory_factory = DiscreteStateTrajectory
        self.segment_factory = DiscreteDwellSegment

    def load_data(self, data_file):
        data_table = pandas.read_csv(data_file, header=0)
        self.trajectory = self.trajectory_factory()
        for segment_data in data_table.itertuples():
            segment_class = str(segment_data[1])
            segment_dwell_time = float(segment_data[2])
            new_segment = self.segment_factory(segment_class,
                                               segment_dwell_time)
            self.trajectory.add_segment(new_segment)

    def get_feature(self):
        return self.trajectory

    def get_target(self):
        return None

    def get_notes(self):
        return []

