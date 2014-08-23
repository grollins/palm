import numpy
from .base.parameter_set import ParameterSet


class CutoffParameterSet(ParameterSet):
    """
    Parameters for a temporal cutoff model. The parameters are
    `tau` and `N`.

    Attributes
    ----------
    parameter_dict : dict
        Parameter values, indexed by parameter names.
    bounds : dict
        Bounds for parameter values (during optimization),
        indexed by parameter names.
    """
    def __init__(self):
        super(CutoffParameterSet, self).__init__()
        self.parameter_dict = {'tau':1.0, 'N':5}
        self.bounds_dict = {'tau':(0.0, 3600.),
                              'N':(None, None)}

    def __str__(self):
        tau = self.get_parameter('tau')
        N = self.get_parameter('N')
        my_str = "%.4f,%d" % (tau, N)
        return my_str

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def __eq__(self, other_param_set):
        if type(self) is type(other_param_set):
            return numpy.array_equal(self.as_array(), other_param_set.as_array())
        else:
            return False

    def set_parameter(self, param_name, param_value):
        if param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        return self.parameter_dict[param_name]

    def as_array(self):
        """
        Converts parameter set to numpy array.
        
        Returns
        -------
        param_array : ndarray
        """
        tau = self.get_parameter('tau')
        N = self.get_parameter('N')
        param_array = numpy.array([tau, N])
        return param_array

    def update_from_array(self, parameter_array):
        """
        Set parameter values from a numpy array. Useful because numpy arrays
        are the input and output type of scipy optimization methods.
        Expected order of parameters in array:
        `tau` and `N`

        Parameters
        ----------
        parameter_array : ndarray
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('tau', parameter_array[0])
        self.set_parameter('N', int(parameter_array[1]))

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        tau_bounds = self.bounds_dict['tau']
        N_bounds = (N, N)
        bounds = [tau_bounds, N_bounds]
        return bounds
