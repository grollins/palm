class DwellTrajectory(object):
    '''
    Based on Qin, Auerbach, and Sachs,
    Proc. R. Soc. Lond. B (1997) 264, 375-383.
    '''
    def __init__(self):
        self.dwell_time_list = []
        self.dwell_class_list = []
        self.cumulative_time_list = [0.0]

    def __len__(self):
        return len(self.dwell_time_list)

    def __str__(self):
        full_str = ""
        for dt, dc in zip(self.dwell_time_list, self.dwell_class_list):
            if dc == "dark":
                dc_id = 0
            else:
                dc_id = 1
            full_str += "%.4e  %d\n" % (dt, dc_id)
        return full_str

    def __iter__(self):
        for dt, dc in zip(self.dwell_time_list, self.dwell_class_list):
            yield dt, dc

    def add_dwell(self, dwell_time, dwell_class):
        self.dwell_time_list.append(dwell_time)
        self.dwell_class_list.append(dwell_class)
        cumulative_time = self.cumulative_time_list[-1] + dwell_time
        self.cumulative_time_list.append(cumulative_time)

    def get_dwell_time(self, k):
        # Qin et al paper uses 1-indexing, which
        # I adopt here.
        return self.dwell_time_list[k-1]

    def get_dwell_class(self, k):
        # Qin et al paper uses 1-indexing, which
        # I adopt here.
        if k > len(self):
            last_class = self.dwell_class_list[-1]
            if last_class == 'bright':
                return 'dark'
            elif last_class == 'dark':
                return 'bright'
            else:
                print "Unknown class", last_class
        else:
            return self.dwell_class_list[k-1]

    def get_cumulative_time(self, k):
        # 1-indexed
        # k=0 is time = 0.0
        # k=1 is time = 0.0 + 1st dwell duration
        return self.cumulative_time_list[k]
