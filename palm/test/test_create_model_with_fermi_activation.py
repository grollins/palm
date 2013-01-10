import nose.tools
import numpy
from palm.blink_parameter_set import SingleDarkParameterSet,\
                                     DoubleDarkParameterSet
from palm.blink_factory import SingleDarkBlinkFactory,\
                               DoubleDarkBlinkFactory
from palm.blink_route_mapper import SingleDarkRouteMapperFactory,\
                                    DoubleDarkRouteMapperFactory

EPSILON = 0.01

@nose.tools.nottest
def compute_fermi_log_ka(time, T, tf):
    numerator = numpy.exp(-(time - tf) / T)
    denominator = ((1 + numerator) * numpy.log(1 + numerator)) * T
    # if denominator < 1e-10:
    #     ka = 0.2
    # else:
    ka = numerator/denominator
    return numpy.log10(ka)

@nose.tools.istest
def activation_rates_should_be_fermi_for_single_dark_state_model():
    sdb_factory = SingleDarkBlinkFactory(fermi_activation=True)
    params = SingleDarkParameterSet()
    T = 1.0
    tf = 200.
    params.set_parameter('N', 1)
    params.set_parameter('fermi_T', T)
    params.set_parameter('fermi_tf', tf)
    m = sdb_factory.create_model(params)
    fermi_T_diff = abs(m.get_parameter('fermi_T') - T)
    fermi_tf_diff = abs(m.get_parameter('fermi_tf') - tf)
    nose.tools.ok_(fermi_T_diff < EPSILON)
    nose.tools.ok_(fermi_tf_diff < EPSILON)
    expected_log_rate = compute_fermi_log_ka(time=tf, T=T, tf=tf)
    num_activation_routes = 0
    for r in m.iter_routes():
        print r.get_label()
        if r.get_label() == 'I->A':
            log_rate = r.log_rate_function(t=tf)
            error_msg = "Expected %.2f, got %.2f" % (expected_log_rate, log_rate)
            rate_diff = abs(expected_log_rate - log_rate)
            nose.tools.ok_(rate_diff < EPSILON, error_msg)
            num_activation_routes += 1
        else:
            continue
    nose.tools.ok_(num_activation_routes > 0, "No activation routes found.")

@nose.tools.istest
def activation_rates_should_be_fermi_for_double_dark_state_model():
    sdb_factory = DoubleDarkBlinkFactory(fermi_activation=True)
    params = DoubleDarkParameterSet()
    T = 1.0
    tf = 200.
    params.set_parameter('N', 1)
    params.set_parameter('fermi_T', T)
    params.set_parameter('fermi_tf', tf)
    m = sdb_factory.create_model(params)
    fermi_T_diff = abs(m.get_parameter('fermi_T') - T)
    fermi_tf_diff = abs(m.get_parameter('fermi_tf') - tf)
    nose.tools.ok_(fermi_T_diff < EPSILON)
    nose.tools.ok_(fermi_tf_diff < EPSILON)
    expected_log_rate = compute_fermi_log_ka(time=tf, T=T, tf=tf)
    num_activation_routes = 0
    for r in m.iter_routes():
        print r.get_label()
        if r.get_label() == 'I->A':
            log_rate = r.log_rate_function(t=tf)
            error_msg = "Expected %.2f, got %.2f" % (expected_log_rate, log_rate)
            rate_diff = abs(expected_log_rate - log_rate)
            nose.tools.ok_(rate_diff < EPSILON, error_msg)
            num_activation_routes += 1
        else:
            continue
    nose.tools.ok_(num_activation_routes > 0, "No activation routes found.")

@nose.tools.istest
def activation_rates_should_be_constant_for_single_dark_state_model():
    sdb_factory = SingleDarkBlinkFactory(fermi_activation=False)
    params = SingleDarkParameterSet()
    expected_log_ka = -4.0
    params.set_parameter('log_ka', expected_log_ka)
    params.set_parameter('N', 1)
    m = sdb_factory.create_model(params)
    num_activation_routes = 0
    for r in m.iter_routes():
        print r.get_label()
        if r.get_label() == 'I->A':
            log_rate = r.log_rate_function(t=0.0)
            error_msg = "Expected %.2f, got %.2f" % (expected_log_ka, log_rate)
            rate_diff = abs(expected_log_ka - log_rate)
            nose.tools.ok_(rate_diff < EPSILON, error_msg)
            num_activation_routes += 1
        else:
            continue
    nose.tools.ok_(num_activation_routes > 0, "No activation routes found.")

@nose.tools.istest
def activation_rates_should_be_constant_for_double_dark_state_model():
    sdb_factory = DoubleDarkBlinkFactory(fermi_activation=False)
    params = DoubleDarkParameterSet()
    expected_log_ka = -4.0
    params.set_parameter('log_ka', expected_log_ka)
    params.set_parameter('N', 1)
    m = sdb_factory.create_model(params)
    num_activation_routes = 0
    for r in m.iter_routes():
        print r.get_label()
        if r.get_label() == 'I->A':
            log_rate = r.log_rate_function(t=0.0)
            error_msg = "Expected %.2f, got %.2f" % (expected_log_ka, log_rate)
            rate_diff = abs(expected_log_ka - log_rate)
            nose.tools.ok_(rate_diff < EPSILON, error_msg)
            num_activation_routes += 1
        else:
            continue
    nose.tools.ok_(num_activation_routes > 0, "No activation routes found.")
