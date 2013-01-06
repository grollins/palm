import numpy
import blink_model
import aggregated_kinetic_model
import base.model_factory
import util
from blink_state_enumerator import SingleDarkStateEnumeratorFactory,\
                                   DoubleDarkStateEnumeratorFactory
from blink_route_mapper import SingleDarkRouteMapperFactory,\
                               DoubleDarkRouteMapperFactory

MAX_A = 1000

class SingleDarkBlinkFactory(base.model_factory.ModelFactory):
    '''
    This factory class creates an aggregated kinetic model with
    the following microstate topology:
    I --> A
    A <--> D
    A --> B
    '''
    def __init__(self, fermi_activation=False):
        self.state_factory = blink_model.SingleDarkState
        self.route_factory = aggregated_kinetic_model.Route
        self.fermi_activation = fermi_activation

    def create_model(self, parameter_set):
        self.parameter_set = parameter_set
        N = self.parameter_set.get_parameter('N')
        state_enumerator_factory = SingleDarkStateEnumeratorFactory(N,
                                                          self.state_factory,
                                                          MAX_A)
        state_enumerator = state_enumerator_factory.create_state_enumerator()
        route_mapper_factory = SingleDarkRouteMapperFactory(self.parameter_set,
                                                            self.route_factory,
                                                            MAX_A,
                                                            self.fermi_activation)
        route_mapper = route_mapper_factory.create_route_mapper()
        new_model = blink_model.BlinkModel(state_enumerator, route_mapper,
                                           self.parameter_set)
        return new_model


class DoubleDarkBlinkFactory(base.model_factory.ModelFactory):
    '''
    This factory class creates an aggregated kinetic model with
    the following microstate topology:
    I --> A
    A <--> D1
    A <--> D2
    A --> B
    '''
    def __init__(self, fermi_activation=False):
        self.state_factory = blink_model.DoubleDarkState
        self.route_factory = aggregated_kinetic_model.Route
        self.fermi_activation = fermi_activation

    def create_model(self, parameter_set):
        self.parameter_set = parameter_set
        N = self.parameter_set.get_parameter('N')
        state_enumerator_factory = DoubleDarkStateEnumeratorFactory(N,
                                                          self.state_factory,
                                                          MAX_A)
        state_enumerator = state_enumerator_factory.create_state_enumerator()
        route_mapper_factory = DoubleDarkRouteMapperFactory(self.parameter_set,
                                                            self.route_factory,
                                                            MAX_A,
                                                            self.fermi_activation)
        route_mapper = route_mapper_factory.create_route_mapper()
        new_model = blink_model.BlinkModel(state_enumerator, route_mapper,
                                           self.parameter_set)
        return new_model
