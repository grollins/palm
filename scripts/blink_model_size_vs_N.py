import palm.blink_factory
import palm.blink_parameter_set

def f(N, model_factory, parameters):
    parameters.set_parameter('N', N)
    model = model_factory.create_model(parameters)
    return model.get_num_states()

def main():
    model_factory = palm.blink_factory.SingleDarkBlinkFactory()
    parameters = palm.blink_parameter_set.SingleDarkParameterSet()
    for i in xrange(1, 50):
        print i, f(i, model_factory, parameters)

if __name__ == '__main__':
    main()
