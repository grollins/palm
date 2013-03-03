import logging
import numpy
import scipy.misc
import time

ALMOST_ZERO = numpy.float64(1e-300)
PORT = 26 * 26**2 + 1 * 26**1 + 13 * 26**0
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

def SetupLogging(Name):
    Logger = logging.getLogger(Name)
    Logger.debug("Module loaded")
    return Logger

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
