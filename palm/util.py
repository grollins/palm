import scipy.misc

ALMOST_ZERO = 1e-300

def n_choose_k(n,k):
    assert n > 0, "%d %d" % (n, k)
    return int(scipy.misc.comb(n, k))
    # return scipy.misc.factorial(n) / (scipy.misc.factorial(k) * scipy.misc.factorial(n-k))

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
