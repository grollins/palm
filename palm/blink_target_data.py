import pandas
import base.target_data
from discrete_state_trajectory import DiscreteStateTrajectory,\
                                      DiscreteDwellSegment

class BlinkTargetData(base.target_data.TargetData):
    """ Expected format
        class,dwell time
        dark,1.5
        bright,0.3
        dark,1.2
        bright,0.1
        .
        .
        .
    """
    def __init__(self):
        super(BlinkTargetData, self).__init__()
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
