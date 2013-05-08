import palm.blink_factory
import palm.blink_parameter_set

def f(N, model_factory, parameters):
    parameters.set_parameter('N', N)
    model = model_factory.create_model(parameters)
    Q = model.build_rate_matrix(time=0.0)
    print len(Q)
    return model.get_num_states(), model.get_num_states('bright'), model.get_num_states('dark')

def main():
    model_factory = palm.blink_factory.SingleDarkBlinkFactory(MAX_A=1)
    parameters = palm.blink_parameter_set.SingleDarkParameterSet()
    print "N,total,bright,dark"
    for i in xrange(1, 81):
        r = f(i, model_factory, parameters)
        print "%d,%d,%d,%d" % (i, r[0], r[1], r[2])

if __name__ == '__main__':
    main()
