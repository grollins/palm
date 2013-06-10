import numpy
import scipy.misc
import time
import random

ALMOST_ZERO = numpy.float64(1e-300)
DATA_TYPE = numpy.float64

def n_choose_k(n,k):
    error_msg = "n must be larger than zero in n_choose_k.\nn=%d, k=%d" % (n, k)
    assert n > 0, error_msg
    return int( round(scipy.misc.comb(n, k)) )

def multichoose(n,k):
    '''
    Generates all combinations of sorting k identical items
    into n separate bins.
    '''
    if k < 0 or n < 0: return "Error"
    if not k: return [[0]*n]
    if not n: return []
    if n == 1: return [[k]]
    return [[0]+val for val in multichoose(n-1,k)] + \
        [[val[0]+1]+val[1:] for val in multichoose(n,k-1)]

def randomize_parameter(parameter_set, parameter_name, lower_bound,
                        upper_bound):
    """
    Change a parameter to a random value within the specified range.
    `parameter_name` in `parameter_set` will be changed in-place to
    the randomly selected value in `[lower_bound, upper_bound)`.

    Parameters
    ----------
    parameter_set : ParameterSet
    parameter_name : string
    lower_bound, upper_bound : float

    Returns
    -------
    parameter_set : ParameterSet
    """
    new_value = random.uniform(lower_bound, upper_bound)
    parameter_set.set_parameter(parameter_name, new_value)
    return parameter_set

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
