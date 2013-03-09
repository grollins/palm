import numpy
import pylab
from palm.blink_route_mapper import SingleDarkRouteMapperFactory,\
                                    DoubleDarkRouteMapperFactory
from palm.blink_parameter_set import SingleDarkParameterSet,\
                                     DoubleDarkParameterSet
from palm.util import ALMOST_ZERO

def local_fermi_fcn(time, T, tf):
    stability_limit = tf + tf/T
    if time > stability_limit:
        time = stability_limit
    numerator = numpy.exp(-(time - tf) / T)
    denominator = ((1 + numerator) * numpy.log(1 + numerator)) * T
    log_ka = numpy.log10(numerator) - numpy.log10(denominator)
    return log_ka, numpy.log10(numerator), numpy.log10(denominator)

def fermi_ka_vs_time():
    params = SingleDarkParameterSet()
    # params = DoubleDarkParameterSet()
    # T = 5.0 # seconds
    # tf = 192.0 # seconds
    T = 20.0 # seconds
    tf = 670.0 # seconds
    params.set_parameter('fermi_T', T)
    params.set_parameter('fermi_tf', tf)
    route_factory = None
    rm_factory = SingleDarkRouteMapperFactory(params, route_factory, max_A=1,
                                              fermi_activation=True)
    # rm_factory = DoubleDarkRouteMapperFactory(params, route_factory, max_A=1)
    rm_fermi_fcn = rm_factory.fermi_log_ka_factory(log_combinatoric_factor=0)

    time_array = numpy.arange(0.0, tf+200, 1.0)
    rm_log_ka_list = []
    local_log_ka_list = []
    numerator_list = []
    denom_list = []
    for t in time_array:
        rm_log_ka_list.append( rm_fermi_fcn(t) )
        local_ka, num, denom = local_fermi_fcn(t, T, tf)
        local_log_ka_list.append( local_ka )
        numerator_list.append(num)
        denom_list.append(denom)
    rm_log_ka_array = numpy.array(rm_log_ka_list)
    local_log_ka_array = numpy.array(local_log_ka_list)

    pylab.plot(time_array - tf, rm_log_ka_array, 'k', lw=3, label='rm')
    pylab.plot(time_array - tf, local_log_ka_array, 'g', lw=3, label='local')
    pylab.show()

def main():
    fermi_ka_vs_time()

if __name__ == '__main__':
    main()
