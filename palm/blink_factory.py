import numpy
from palm.base.model_factory import ModelFactory
from palm.blink_model import BlinkModel
from palm.blink_state_enumerator import SingleDarkState, DoubleDarkState,\
                                        SingleDarkStateEnumeratorFactory,\
                                        DoubleDarkStateEnumeratorFactory
from palm.blink_route_mapper import Route, SingleDarkRouteMapperFactory,\
                                    DoubleDarkRouteMapperFactory


class SingleDarkBlinkFactory(ModelFactory):
    '''
    This factory class creates an aggregated kinetic model with
    the following microstate topology:
    I --> A
    A <--> D
    A --> B
    '''
    def __init__(self, fermi_activation=False, MAX_A=10):
        self.state_factory = SingleDarkState
        self.route_factory = Route
        self.fermi_activation = fermi_activation
        self.MAX_A = MAX_A

    def create_model(self, parameter_set):
        self.parameter_set = parameter_set
        N = self.parameter_set.get_parameter('N')
        state_enumerator_factory = SingleDarkStateEnumeratorFactory(
                                        N, self.state_factory, self.MAX_A)
        state_enumerator = state_enumerator_factory.create_state_enumerator()
        route_mapper_factory = SingleDarkRouteMapperFactory(
                                parameter_set=self.parameter_set,
                                route_factory=self.route_factory,
                                max_A=self.MAX_A)
        route_mapper = route_mapper_factory.create_route_mapper()
        new_model = BlinkModel(state_enumerator, route_mapper,
                               self.parameter_set, self.fermi_activation)
        return new_model


class DoubleDarkBlinkFactory(ModelFactory):
    '''
    This factory class creates an aggregated kinetic model with
    the following microstate topology:
    I --> A
    A <--> D1
    A <--> D2
    A --> B
    '''
    def __init__(self, fermi_activation=False, MAX_A=10):
        self.state_factory = DoubleDarkState
        self.route_factory = Route
        self.fermi_activation = fermi_activation
        self.MAX_A = MAX_A

    def create_model(self, parameter_set):
        self.parameter_set = parameter_set
        N = self.parameter_set.get_parameter('N')
        state_enumerator_factory = DoubleDarkStateEnumeratorFactory(
                                        N, self.state_factory, self.MAX_A)
        state_enumerator = state_enumerator_factory.create_state_enumerator()
        route_mapper_factory = DoubleDarkRouteMapperFactory(
                                parameter_set=self.parameter_set,
                                route_factory=self.route_factory,
                                max_A=self.MAX_A)
        route_mapper = route_mapper_factory.create_route_mapper()
        new_model = BlinkModel(state_enumerator, route_mapper,
                                           self.parameter_set)
        return new_model
