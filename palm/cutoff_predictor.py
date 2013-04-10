import numpy
from palm.base.data_predictor import DataPredictor
from palm.base.prediction import Prediction

class CutoffPrediction(Prediction):
    """docstring for CutoffPrediction"""
    def __init__(self, num_bundles):
        super(CutoffPrediction, self).__init__()
        self.num_bundles = num_bundles
    def __str__(self):
        return str(self.num_bundles)
    def __eq__(self, other_cutoff_prediction):
        return self.num_bundles == other_cutoff_prediction.num_bundles
    def as_array(self):
        return numpy.array([self.num_bundles,])

class CutoffPredictor(DataPredictor):
    """Dark dwells longer than tau signify activation events.
       Dark dwells below tau signify blinking events.
       Blinking events are referred to as bundles here, since
       the consecutive bright dwells are bundled together and
       attributed to the same fluorophore.
    """
    def __init__(self, tau):
        super(CutoffPredictor, self).__init__()
        self.tau = tau  # seconds

    def predict_data(self, trajectory):
        return self.count_bundles(trajectory)

    def count_bundles(self, trajectory):
        dark_dwell_list = []
        # loop through trajectory segments, save dark dwell durations
        for segment_number, segment in enumerate(trajectory):
            if segment_number == 0:
                assert segment.get_class() == 'dark'
                continue
            elif segment_number == 1:
                assert segment.get_class() == 'bright'
            elif segment_number == (len(trajectory) - 1):
                assert segment.get_class() == 'dark'
                continue
            else:
                if segment.get_class() == 'dark':
                    dark_dwell_list.append(segment.get_duration())
                else:
                    continue
        dd_array = numpy.array(dark_dwell_list)
        num_bundles = len(numpy.where(dd_array > self.tau)[0])
        num_bundles += 1  # trajectory starts with a dark dwell and an activation event
        return CutoffPrediction(num_bundles)
