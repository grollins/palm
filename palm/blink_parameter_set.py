import numpy
from .base.parameter_set import ParameterSet


class SingleDarkParameterSet(ParameterSet):
    """
    Parameters for a blink model with one dark state. The parameters are
    `log_ka`, `log_kr`, `log_kd`, `log_kb`, `N`, `fermi_T`, and `fermi_tf`.

    Attributes
    ----------
    parameter_dict : dict
        Parameter values, indexed by parameter names.
    bounds : dict
        Bounds for parameter values (during optimization),
        indexed by parameter names.
    """
    def __init__(self):
        super(SingleDarkParameterSet, self).__init__()
        self.parameter_dict = {'log_ka':-1.0, 'log_kd':-1.0,
                               'log_kr':-1.0, 'log_kb':-1.0, 'N':5,
                               'fermi_T':5.0, 'fermi_tf':192.}
        self.bounds_dict = {'log_ka':(None, None),
                            'log_kd':(None, None),
                            'log_kr':(None, None),
                            'log_kb':(None, None) }

    def __str__(self):
        log_ka = self.get_parameter('log_ka')
        log_kd = self.get_parameter('log_kd')
        log_kr = self.get_parameter('log_kr')
        log_kb = self.get_parameter('log_kb')
        my_str = "%.4f,%.4f,%.4f,%.4f" % (log_ka, log_kd, log_kr, log_kb)
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
        log_ka = self.get_parameter('log_ka')
        log_kd = self.get_parameter('log_kd')
        log_kr = self.get_parameter('log_kr')
        log_kb = self.get_parameter('log_kb')
        N = self.get_parameter('N')
        fermi_T = self.get_parameter('fermi_T')
        fermi_tf = self.get_parameter('fermi_tf')
        param_array = numpy.array([log_ka, log_kd, log_kr, log_kb, N,
                                   fermi_T, fermi_tf])
        return param_array

    def update_from_array(self, parameter_array):
        """
        Set parameter values from a numpy array. Useful because numpy arrays
        are the input and output type of scipy optimization methods.
        Expected order of parameters in array:
        `log_ka`, `log_kd`, `log_kr`, `log_kb`, `N`, `fermi_T`, `fermi_tf`

        Parameters
        ----------
        parameter_array : ndarray
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_ka', parameter_array[0])
        self.set_parameter('log_kd', parameter_array[1])
        self.set_parameter('log_kr', parameter_array[2])
        self.set_parameter('log_kb', parameter_array[3])
        self.set_parameter('N', int(parameter_array[4]))
        self.set_parameter('fermi_T', parameter_array[5])
        self.set_parameter('fermi_tf', parameter_array[6])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        log_ka_bounds = self.bounds_dict['log_ka']
        log_kd_bounds = self.bounds_dict['log_kd']
        log_kr_bounds = self.bounds_dict['log_kr']
        log_kb_bounds = self.bounds_dict['log_kb']
        N = self.get_parameter('N')
        N_bounds = (N, N)
        fermi_T = self.get_parameter('fermi_T')
        fermi_T_bounds = (fermi_T, fermi_T)
        fermi_tf = self.get_parameter('fermi_tf')
        fermi_tf_bounds = (fermi_tf, fermi_tf)
        bounds = [log_ka_bounds, log_kd_bounds,
                  log_kr_bounds, log_kb_bounds,
                  N_bounds, fermi_T_bounds, fermi_tf_bounds]
        return bounds


class DoubleDarkParameterSet(ParameterSet):
    """
    Parameters for a blink model with one dark state. The parameters are
    `log_ka`, `log_kr`, `log_kr_diff`, `log_kd1`, `log_kd2`, `log_kb`, `N`,
    `fermi_T`, and `fermi_tf`.

    Attributes
    ----------
    parameter_dict : dict
        Parameter values, indexed by parameter names.
    bounds : dict
        Bounds for parameter values (during optimization),
        indexed by parameter names.
    """
    def __init__(self):
        super(DoubleDarkParameterSet, self).__init__()
        self.parameter_dict = {'log_ka':-1.0, 'log_kd1':-1.0,
                               'log_kr1':-1.0, 'log_kd2':-2.0,
                               'log_kr_diff':-2.0, 'log_kb':-1.0, 'N':5,
                               'fermi_T':5.0, 'fermi_tf':192.}
        self.bounds_dict = {'log_ka':(None, None),
                            'log_kd1':(None, None),
                            'log_kr1':(None, None),
                            'log_kd2':(None, None),
                            'log_kr_diff':(None, None),
                            'log_kb':(None, None)}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

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
        log_ka = self.get_parameter('log_ka')
        log_kd1 = self.get_parameter('log_kd1')
        log_kr1 = self.get_parameter('log_kr1')
        log_kd2 = self.get_parameter('log_kd2')
        log_kr_diff = self.get_parameter('log_kr_diff')
        log_kb = self.get_parameter('log_kb')
        N = self.get_parameter('N')
        fermi_T = self.get_parameter('fermi_T')
        fermi_tf = self.get_parameter('fermi_tf')
        return numpy.array([log_ka, log_kd1, log_kr1, log_kd2, log_kr_diff,
                            log_kb, N, fermi_T, fermi_tf])

    def update_from_array(self, parameter_array):
        """
        Set parameter values from a numpy array. Useful because numpy arrays
        are the input and output type of scipy optimization methods.
        Expected order of parameters in array:
        `log_ka`, `log_kd1`, `log_kr1`, `log_kd2`, `log_kr_diff`,
        `log_kb`, `N`, `fermi_T`, `fermi_tf`.

        Parameters
        ----------
        parameter_array : ndarray
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_ka', parameter_array[0])
        self.set_parameter('log_kd1', parameter_array[1])
        self.set_parameter('log_kr1', parameter_array[2])
        self.set_parameter('log_kd2', parameter_array[3])
        self.set_parameter('log_kr_diff', parameter_array[4])
        self.set_parameter('log_kb', parameter_array[5])
        self.set_parameter('N', int(parameter_array[6]))
        self.set_parameter('fermi_T', parameter_array[7])
        self.set_parameter('fermi_tf', parameter_array[8])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        log_ka_bounds = self.bounds_dict['log_ka']
        log_kd1_bounds = self.bounds_dict['log_kd1']
        log_kr1_bounds = self.bounds_dict['log_kr1']
        log_kd2_bounds = self.bounds_dict['log_kd2']
        log_kr_diff_bounds = self.bounds_dict['log_kr_diff']
        log_kb_bounds = self.bounds_dict['log_kb']
        N = self.get_parameter('N')
        N_bounds = (N, N)
        fermi_T = self.get_parameter('fermi_T')
        fermi_T_bounds = (fermi_T, fermi_T)
        fermi_tf = self.get_parameter('fermi_tf')
        fermi_tf_bounds = (fermi_tf, fermi_tf)
        bounds = [log_ka_bounds, log_kd1_bounds,
                  log_kr1_bounds, log_kd2_bounds,
                  log_kr_diff_bounds, log_kb_bounds,
                  N_bounds, fermi_T_bounds, fermi_tf_bounds]


class ConnectedDarkParameterSet(ParameterSet):
    """
    Parameters for a blink model with one dark state. The parameters are
    `log_ka`, `log_kr1`, `log_kr2`, `log_kd1`, `log_kd2`, `log_kb`, `N`,
    `fermi_T`, and `fermi_tf`.

    Attributes
    ----------
    parameter_dict : dict
        Parameter values, indexed by parameter names.
    bounds : dict
        Bounds for parameter values (during optimization),
        indexed by parameter names.
    """
    def __init__(self):
        super(ConnectedDarkParameterSet, self).__init__()
        self.parameter_dict = {'log_ka':-1.0, 'log_kd1':-1.0,
                               'log_kr1':-1.0, 'log_kd2':-2.0,
                               'log_kr2':-2.0, 'log_kb':-1.0, 'N':5,
                               'fermi_T':5.0, 'fermi_tf':192.}
        self.bounds_dict = {'log_ka':(None, None),
                            'log_kd1':(None, None),
                            'log_kr1':(None, None),
                            'log_kd2':(None, None),
                            'log_kr2':(None, None),
                            'log_kb':(None, None)}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

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
        log_ka = self.get_parameter('log_ka')
        log_kd1 = self.get_parameter('log_kd1')
        log_kr1 = self.get_parameter('log_kr1')
        log_kd2 = self.get_parameter('log_kd2')
        log_kr2 = self.get_parameter('log_kr2')
        log_kb = self.get_parameter('log_kb')
        N = self.get_parameter('N')
        fermi_T = self.get_parameter('fermi_T')
        fermi_tf = self.get_parameter('fermi_tf')
        return numpy.array([log_ka, log_kd1, log_kr1, log_kd2, log_kr2,
                            log_kb, N, fermi_T, fermi_tf])

    def update_from_array(self, parameter_array):
        """
        Set parameter values from a numpy array. Useful because numpy arrays
        are the input and output type of scipy optimization methods.
        Expected order of parameters in array:
        `log_ka`, `log_kd1`, `log_kr1`, `log_kd2`, `log_kr2`,
        `log_kb`, `N`, `fermi_T`, `fermi_tf`.

        Parameters
        ----------
        parameter_array : ndarray
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_ka', parameter_array[0])
        self.set_parameter('log_kd1', parameter_array[1])
        self.set_parameter('log_kr1', parameter_array[2])
        self.set_parameter('log_kd2', parameter_array[3])
        self.set_parameter('log_kr2', parameter_array[4])
        self.set_parameter('log_kb', parameter_array[5])
        self.set_parameter('N', int(parameter_array[6]))
        self.set_parameter('fermi_T', parameter_array[7])
        self.set_parameter('fermi_tf', parameter_array[8])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        log_ka_bounds = self.bounds_dict['log_ka']
        log_kd1_bounds = self.bounds_dict['log_kd1']
        log_kr1_bounds = self.bounds_dict['log_kr1']
        log_kd2_bounds = self.bounds_dict['log_kd2']
        log_kr2_bounds = self.bounds_dict['log_kr2']
        log_kb_bounds = self.bounds_dict['log_kb']
        N = self.get_parameter('N')
        N_bounds = (N, N)
        fermi_T = self.get_parameter('fermi_T')
        fermi_T_bounds = (fermi_T, fermi_T)
        fermi_tf = self.get_parameter('fermi_tf')
        fermi_tf_bounds = (fermi_tf, fermi_tf)
        bounds = [log_ka_bounds, log_kd1_bounds,
                  log_kr1_bounds, log_kd2_bounds,
                  log_kr2_bounds, log_kb_bounds,
                  N_bounds, fermi_T_bounds, fermi_tf_bounds]
