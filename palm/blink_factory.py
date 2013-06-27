import numpy
from palm.base.model_factory import ModelFactory
from palm.blink_model import BlinkModel
from palm.blink_state_enumerator import SingleDarkState, DoubleDarkState,\
                                        SingleDarkStateEnumeratorFactory,\
                                        DoubleDarkStateEnumeratorFactory
from palm.blink_route_mapper import Route, SingleDarkRouteMapperFactory,\
                                    DoubleDarkRouteMapperFactory,\
                                    ConnectedDarkRouteMapperFactory


class SingleDarkBlinkFactory(ModelFactory):
    '''
    This factory class creates an aggregated kinetic model with
    one dark state: (insert image)

    Parameters
    ----------
    fermi_activation : bool, optional
        Whether the activation rates vary with time.
    MAX_A : int, optional
        Number of fluorophores that can be simultaneously active.

    Attributes
    ----------
    state_factory : class
        Factory class for State objects.
    route_factory : class
        Factory class for Route objects.
    '''
    def __init__(self, fermi_activation=False, MAX_A=10):
        self.state_factory = SingleDarkState
        self.route_factory = Route
        self.fermi_activation = fermi_activation
        self.MAX_A = MAX_A

    def create_model(self, parameter_set):
        """
        Creates a new BlinkModel with one dark state.

        Parameters
        ----------
        parameter_set : SingleDarkParameterSet
            Parameters for a model with one dark state.

        Returns
        -------
        new_model : BlinkModel
        """
        N = parameter_set.get_parameter('N')
        state_enumerator_factory = SingleDarkStateEnumeratorFactory(
                                        N, self.state_factory, self.MAX_A)
        state_enumerator = state_enumerator_factory.create_state_enumerator()
        route_mapper_factory = SingleDarkRouteMapperFactory(
                                parameter_set=parameter_set,
                                route_factory=self.route_factory,
                                max_A=self.MAX_A)
        route_mapper = route_mapper_factory.create_route_mapper()
        new_model = BlinkModel(state_enumerator, route_mapper,
                               parameter_set, self.fermi_activation)
        return new_model


class DoubleDarkBlinkFactory(ModelFactory):
    '''
    This factory class creates an aggregated kinetic model with
    two, unconnected dark states: (insert image)

    Parameters
    ----------
    fermi_activation : bool, optional
        Whether the activation rates vary with time.
    MAX_A : int, optional
        Number of fluorophores that can be simultaneously active.

    Attributes
    ----------
    state_factory : class
        Factory class for State objects.
    route_factory : class
        Factory class for Route objects.
    '''
    def __init__(self, fermi_activation=False, MAX_A=10):
        self.state_factory = DoubleDarkState
        self.route_factory = Route
        self.fermi_activation = fermi_activation
        self.MAX_A = MAX_A

    def create_model(self, parameter_set):
        """
        Creates a new BlinkModel with two dark states.

        Parameters
        ----------
        parameter_set : DoubleDarkParameterSet
            Parameters for a model with one dark state.

        Returns
        -------
        new_model : BlinkModel
        """
        N = parameter_set.get_parameter('N')
        state_enumerator_factory = DoubleDarkStateEnumeratorFactory(
                                        N, self.state_factory, self.MAX_A)
        state_enumerator = state_enumerator_factory.create_state_enumerator()
        route_mapper_factory = DoubleDarkRouteMapperFactory(
                                parameter_set=parameter_set,
                                route_factory=self.route_factory,
                                max_A=self.MAX_A)
        route_mapper = route_mapper_factory.create_route_mapper()
        new_model = BlinkModel(state_enumerator, route_mapper,
                               parameter_set)
        return new_model


class ConnectedDarkBlinkFactory(ModelFactory):
    '''
    This factory class creates an aggregated kinetic model with
    two, connected dark states: (insert image)

    Parameters
    ----------
    fermi_activation : bool, optional
        Whether the activation rates vary with time.
    MAX_A : int, optional
        Number of fluorophores that can be simultaneously active.

    Attributes
    ----------
    state_factory : class
        Factory class for State objects.
    route_factory : class
        Factory class for Route objects.
    '''
    def __init__(self, fermi_activation=False, MAX_A=10):
        self.state_factory = DoubleDarkState
        self.route_factory = Route
        self.fermi_activation = fermi_activation
        self.MAX_A = MAX_A

    def create_model(self, parameter_set):
        """
        Creates a new BlinkModel with two dark states.

        Parameters
        ----------
        parameter_set : ConnectedDarkParameterSet
            Parameters for a model with one dark state.

        Returns
        -------
        new_model : BlinkModel
        """
        N = parameter_set.get_parameter('N')
        state_enumerator_factory = DoubleDarkStateEnumeratorFactory(
                                        N, self.state_factory, self.MAX_A)
        state_enumerator = state_enumerator_factory.create_state_enumerator()
        route_mapper_factory = ConnectedDarkRouteMapperFactory(
                                parameter_set=parameter_set,
                                route_factory=self.route_factory,
                                max_A=self.MAX_A)
        route_mapper = route_mapper_factory.create_route_mapper()
        new_model = BlinkModel(state_enumerator, route_mapper,
                               parameter_set)
        return new_model
