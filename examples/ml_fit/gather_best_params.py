import sys
import os.path
import glob
from palm.blink_target_data import BlinkCollectionTargetData
from palm.parameter_set_distribution import ParamSetDistFactory,\
                                            ParameterSetDistribution
from palm.blink_parameter_set import SingleDarkParameterSet

def load_previous_run(prev_run_filename):
    prev_psd = ParameterSetDistribution()
    prev_psd.load_from_file(prev_run_filename)
    return prev_psd

def get_best_series(prev_psd):
    best_scoring_series = prev_psd.get_best_params_as_series()
    return best_scoring_series

def main():
    try:
        bootstrap_size = int(sys.argv[1])
    except IndexError:
        print "\nExample usage:\npython gather_best_params.py <bootstrap_size>\n" \
              "\twhere <bootstrap_size> is an integer that corresponds to\n" \
              "\tthe value of BOOTSTRAP_SIZE in bootstrap_ml_fit.py\n"
        raise
    psd_factory = ParamSetDistFactory()
    prev_run_files = glob.glob("params/%03d_*_params.pkl") % bootstrap_size
    for p in prev_run_files:
        prev_psd = load_previous_run(p)
        best_scoring_series = get_best_series(prev_psd)
        N = best_scoring_series['N']
        log_ka = best_scoring_series['log_ka']
        log_kd = best_scoring_series['log_kd']
        log_kr = best_scoring_series['log_kr']
        log_kb = best_scoring_series['log_kb']
        this_param_set = SingleDarkParameterSet()
        this_param_set.set_parameter('N',  N)
        this_param_set.set_parameter('log_ka',  log_ka)
        this_param_set.set_parameter('log_kd',  log_kd)
        this_param_set.set_parameter('log_kr',  log_kr)
        this_param_set.set_parameter('log_kb',  log_kb)
        psd_factory.add_parameter_set(this_param_set)
    psd = psd_factory.make_psd()

    # save results from this run
    param_pkl_stream = os.path.join(
                        './params', "best.pkl")
    psd.save_to_file(param_pkl_stream)
    print "Wrote %s" % (param_pkl_stream)
    param_html_stream = os.path.join(
                        './params', "best.html")
    psd.to_html(param_html_stream)
    print "Wrote %s" % (param_html_stream)

if __name__ == '__main__':
    main()

