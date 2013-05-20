import numpy
from palm.aggregated_kinetic_model import AggregatedKineticModel
from palm.probability_vector import make_prob_vec_from_state_ids
from palm.state_collection import StateIDCollection

class BlinkModel(AggregatedKineticModel):
    """
    BlinkModel is an AggregatedKineticModel. Two observation classes
    are expected:
    1. dark (no fluorescence detected)
    2. bright (fluorescence detected)

    Attributes
    ----------
    all_inactive_state_id : string
    all_bleached_state_id : string

    Parameters
    ----------
    state_enumerator : callable f()
        Generates a StateCollection for the model.
    route_mapper : callable f(state_collection)
        Generates a RouteCollection for the model.
    parameter_set : ParameterSet
    fermi_activation : bool, optional
        Whether the activation rates vary with time.
    """
    def __init__(self, state_enumerator, route_mapper, parameter_set,
                 fermi_activation=False):
        super(BlinkModel, self).__init__(state_enumerator, route_mapper,
                                          parameter_set, fermi_activation)
        self.all_inactive_state_id = self.initial_state_id
        self.all_bleached_state_id = self.final_state_id

    def get_initial_probability_vector(self):
        """
        Creates a vector with probability density localized to
        the all-inactive state.

        Returns
        -------
        initial_prob_vec : ProbabilityVector
        """
        dark_state_id_collection = self.state_ids_by_class_dict['dark']
        initial_prob_vec = make_prob_vec_from_state_ids(dark_state_id_collection)
        initial_prob_vec.set_state_probability(self.all_inactive_state_id, 1.0)
        return initial_prob_vec

    def get_final_probability_vector(self):
        """
        Creates a vector with probability density localized to
        the all-photobleached state.

        Returns
        -------
        final_prob_vec : ProbabilityVector
        """
        dark_state_id_collection = self.state_ids_by_class_dict['dark']
        final_prob_vec = make_prob_vec_from_state_ids(dark_state_id_collection)
        final_prob_vec.set_state_probability(self.all_bleached_state_id, 1.0)
        return final_prob_vec
