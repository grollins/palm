import numpy

PARAM_NAME_DICT = {'ka':'log_ka', 'kd':'log_kd', 'kd1':'log_kd1',
                   'kd2':'log_kd2', 'kr':'log_kr', 'kr1':'log_kr1',
                   'kb':'log_kb', 'A_to_B':'log_k1', 'B_to_A':'log_k2'}

def rate_from_rate_id(rate_id, t, parameter_set, fermi_activation):
    if rate_id == 'ka' and fermi_activation:
        T = parameter_set.get_parameter('fermi_T')
        tf = parameter_set.get_parameter('fermi_tf')
        stability_limit = tf + tf/T
        if t > stability_limit:
            t = stability_limit
        numerator = numpy.exp(-(t - tf) / T)
        denominator = ((1 + numerator) * numpy.log(1 + numerator)) * T
        ka = numerator / denominator
        return ka
    elif rate_id == 'kr2':
        log_kr_diff = parameter_set.get_parameter('log_kr_diff')
        log_kr1 = parameter_set.get_parameter('log_kr1')
        log_kr2 = log_kr1 + log_kr_diff
        kr2 = 10**log_kr2
        return kr2
    else:
        param_name = PARAM_NAME_DICT[rate_id]
        log_rate = parameter_set.get_parameter(param_name)
        rate = 10**log_rate
        return rate
