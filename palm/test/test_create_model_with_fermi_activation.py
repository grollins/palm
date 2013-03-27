import nose.tools
import numpy
from palm.blink_parameter_set import SingleDarkParameterSet,\
                                     DoubleDarkParameterSet
from palm.blink_factory import SingleDarkBlinkFactory,\
                               DoubleDarkBlinkFactory
from palm.blink_route_mapper import SingleDarkRouteMapperFactory,\
                                    DoubleDarkRouteMapperFactory
from palm.rate_fcn import rate_from_rate_id

EPSILON = 0.01

@nose.tools.nottest
def compute_fermi_ka(time, T, tf):
    numerator = numpy.exp(-(time - tf) / T)
    denominator = ((1 + numerator) * numpy.log(1 + numerator)) * T
    # if denominator < 1e-10:
    #     ka = 0.2
    # else:
    ka = numerator/denominator
    return ka

@nose.tools.istest
def activation_rates_should_be_fermi_for_single_dark_state_model():
    sdb_factory = SingleDarkBlinkFactory(fermi_activation=True)
    params = SingleDarkParameterSet()
    T = 1.0
    tf = 200.
    params.set_parameter('N', 3)
    params.set_parameter('fermi_T', T)
    params.set_parameter('fermi_tf', tf)
    m = sdb_factory.create_model(params)
    fermi_T_diff = abs(m.get_parameter('fermi_T') - T)
    fermi_tf_diff = abs(m.get_parameter('fermi_tf') - tf)
    nose.tools.ok_(fermi_T_diff < EPSILON)
    nose.tools.ok_(fermi_tf_diff < EPSILON)
    Q = m.build_rate_matrix(time=tf)
    for r_id, r in m.route_collection.iter_routes():
        rate_id = r['rate_id']
        if rate_id == 'ka':
            start_state = r['start_state']
            end_state = r['end_state']
            mult = r['multiplicity']
            this_rate = Q.get_rate(start_state, end_state)
            log_rate = numpy.log10(this_rate)
            expected_rate = mult * compute_fermi_ka(time=tf, T=T, tf=tf)
            expected_log_rate = numpy.log10(expected_rate)
            error_msg = "Expected %.2f, got %.2f for %s" %\
                        (expected_log_rate, log_rate, r_id)
            rate_diff = abs(expected_log_rate - log_rate)
            nose.tools.ok_(rate_diff < EPSILON, error_msg)
            print error_msg
        else:
            continue

@nose.tools.istest
def activation_rates_should_be_constant_for_single_dark_state_model():
    sdb_factory = SingleDarkBlinkFactory(fermi_activation=False)
    params = SingleDarkParameterSet()
    log_ka = -4.0
    params.set_parameter('log_ka', log_ka)
    params.set_parameter('N', 3)
    m = sdb_factory.create_model(params)
    Q = m.build_rate_matrix(time=0.0)
    for r_id, r in m.route_collection.iter_routes():
        rate_id = r['rate_id']
        if rate_id == 'ka':
            start_state = r['start_state']
            end_state = r['end_state']
            mult = r['multiplicity']
            this_rate = Q.get_rate(start_state, end_state)
            log_rate = numpy.log10(this_rate)
            expected_rate = mult * 10**log_ka
            expected_log_rate = numpy.log10(expected_rate)
            error_msg = "Expected %.2f, got %.2f for %s" %\
                        (expected_log_rate, log_rate, r_id)
            rate_diff = abs(expected_log_rate - log_rate)
            nose.tools.ok_(rate_diff < EPSILON, error_msg)
            print error_msg
        else:
            continue

