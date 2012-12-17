import numpy
import aggregated_kinetic_model

class BlinkState(object):
    def __init__(self, id_str, I, A, D, B, observation_class):
        self.id = id_str
        self.I = I
        self.A = A
        self.D = D
        self.B = B
        self.observation_class = observation_class
        self.is_start_state = False
        self.is_end_state = False

    def __str__(self):
        return "%s %s" % (self.id, self.observation_class)

    def as_array(self):
        return numpy.array([self.I, self.A, self.D, self.B])

    def get_id(self):
        return self.id

    def get_class(self):
        return self.observation_class


class BlinkModel(aggregated_kinetic_model.AggregatedKineticModel):
    def __init__(self, state_enumerator, route_mapper, parameter_set):
        super(BlinkModel, self).__init__(state_enumerator, route_mapper,
                                          parameter_set)

    def get_initial_population_array(self):
        initial_population_array = numpy.zeros([1, self.get_num_states('dark')])
        initial_population_array[0,0] = 1.0
        return initial_population_array

