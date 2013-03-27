import numpy
from palm.aggregated_kinetic_model import AggregatedKineticModel
from palm.probability_vector import make_prob_vec_from_state_ids
from palm.state_collection import StateIDCollection

class BlinkModel(AggregatedKineticModel):
    '''
    BlinkModel is an AggregatedKineticModel. Two observation classes
    are expected:
    1. dark (no fluorescence detected)
    2. bright (fluorescence detected)
    '''
    def __init__(self, state_enumerator, route_mapper, parameter_set,
                 fermi_activation=False):
        super(BlinkModel, self).__init__(state_enumerator, route_mapper,
                                          parameter_set, fermi_activation)
        self.all_inactive_state_id = self.initial_state_id
        self.all_bleached_state_id = self.final_state_id

    def get_initial_probability_vector(self):
        dark_state_id_collection = self.state_ids_by_class_dict['dark']
        initial_prob_vec = make_prob_vec_from_state_ids(dark_state_id_collection)
        initial_prob_vec.set_state_probability(self.all_inactive_state_id, 1.0)
        return initial_prob_vec
