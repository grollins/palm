import nose.tools
from palm.parameter_set_distribution import ParameterSetDistribution
from palm.parameter_set_distribution import ParamSetDistFactory
from palm.blink_parameter_set import SingleDarkParameterSet

@nose.tools.istest
def param_set_distribution_can_be_saved_and_loaded():
    parameter_set1 = SingleDarkParameterSet()
    parameter_set2 = SingleDarkParameterSet()
    parameter_set3 = SingleDarkParameterSet()
    psd_factory = ParamSetDistFactory()
    psd_factory.add_parameter_set(parameter_set1)
    psd_factory.add_parameter_set(parameter_set2)
    psd_factory.add_parameter_set(parameter_set3)
    psd = psd_factory.make_psd()
    psd.save_to_file('temp_psd.pkl')
    reloaded_psd = ParameterSetDistribution()
    reloaded_psd.load_from_file('temp_psd.pkl')
    print len(psd), len(reloaded_psd)

@nose.tools.istest
def select_param_set_and_find_minimum_value():
    parameter_set1 = SingleDarkParameterSet()
    parameter_set2 = SingleDarkParameterSet()
    parameter_set3 = SingleDarkParameterSet()
    psd_factory = ParamSetDistFactory()
    psd_factory.add_parameter_set(parameter_set1)
    psd_factory.add_parameter('score', 0.0)
    psd_factory.add_parameter('id', 'a')
    psd_factory.add_parameter_set(parameter_set2)
    psd_factory.add_parameter('score', 1.3)
    psd_factory.add_parameter('id', 'c')
    psd_factory.add_parameter_set(parameter_set3)
    psd_factory.add_parameter('score', -1.0)
    psd_factory.add_parameter('id', 'a')
    psd = psd_factory.make_psd()
    sub_df = psd.select_param_sets('id', 'a')
    minimum_score_row_index = sub_df['score'].idxmin()
    minimum_score_row = sub_df.ix[minimum_score_row_index]
    print psd
    print sub_df
    print minimum_score_row_index
    print minimum_score_row
    new_psd_factory = ParamSetDistFactory()
    new_psd_factory.add_parameters_from_data_series(minimum_score_row)
    minimum_psd = new_psd_factory.make_psd()
    minimum_psd.sort_index('N', is_ascending=True)
    print minimum_psd
