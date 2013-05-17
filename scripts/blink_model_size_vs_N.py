import palm.blink_factory
import palm.blink_parameter_set

def f(N, model_factory, parameters):
    parameters.set_parameter('N', N)
    parameters.set_parameter('log_ka', -0.3)
    parameters.set_parameter('log_kd', 1.0)
    parameters.set_parameter('log_kr', -1.0)
    parameters.set_parameter('log_kb', 0.3)

    model = model_factory.create_model(parameters)
    Q = model.build_rate_matrix(time=0.0)
    Q_bb = model.get_submatrix(Q, 'bright', 'bright')
    Q_bb.data_frame *= 0.05
    Q_bb.print_non_zero_entries()
    #print len(Q)
    return model.get_num_states(), model.get_num_states('bright'),\
           model.get_num_states('dark'), Q_bb.compute_sparsity(),\
           Q_bb.compute_max_element_magnitude(), Q_bb.compute_norm()

def main():
    model_factory = palm.blink_factory.SingleDarkBlinkFactory(MAX_A=10)
    parameters = palm.blink_parameter_set.SingleDarkParameterSet()
    print "N,total,bright,dark"
    for i in xrange(1, 81):
        r = f(i, model_factory, parameters)
        print "%d,%d,%d,%d,%.3f,%.2f" % (i, r[0], r[1], r[2], r[3], r[4])
        print r[5]

if __name__ == '__main__':
    main()
