import sys
import os.path
from palm.bootstrap_selector import BootstrapSelector
from palm.blink_target_data import BlinkCollectionTargetData
from palm.parameter_set_distribution import ParamSetDistFactory
from opt_fcn import run_optimization

DIRECTORY_FILE = os.path.abspath('./traj_paths.txt')
BOOTSTRAP_SIZE = 20  # number of trajectories per maximum likelihood fit

def make_bs_file(traj_data, bs_selector, size, filename):
    resampled_traj_data = bs_selector.select_data(traj_data, size=size)
    num_segments = resampled_traj_data.get_total_number_of_trajectory_segments()
    with open(filename, 'w') as f:
        for p in resampled_traj_data.get_paths():
            f.write("%s\n" % p)
    return num_segments

def main():
    id_str = sys.argv[1]  # identifier for this maximum likelihood fit
    traj_data = BlinkCollectionTargetData()
    traj_data.load_data(DIRECTORY_FILE)
    bs_selector = BootstrapSelector()
    bs_file_list = []
    filename = os.path.abspath(
                './bootstrap_files/bootstrap_%03d_%03d.txt' % (BOOTSTRAP_SIZE, int(id_str)))
    num_segments = make_bs_file(traj_data, bs_selector, BOOTSTRAP_SIZE,
                                filename)
    psd_factory = ParamSetDistFactory()

    for N in xrange(1, 6, 1):
        print N
        r = run_optimization(N, filename)
        this_N, this_score, this_param_set = r
        psd_factory.add_parameter_set(this_param_set)
        psd_factory.add_parameter('id_str', id_str)
        psd_factory.add_parameter('score', this_score)
        psd_factory.add_parameter('num segments', num_segments)
        psd_factory.add_parameter('num trajs', BOOTSTRAP_SIZE)
        psd = psd_factory.make_psd()
        psd.sort_index('N', is_ascending=True)

        # save results
        param_pkl_stream = os.path.join(
                            './params', "%03d_%03d_params.pkl" %\
                            (BOOTSTRAP_SIZE, int(id_str)))
        psd.save_to_file(param_pkl_stream)
        print "Wrote %s" % (param_pkl_stream)
        param_html_stream = os.path.join(
                            './params', "%03d_%03d_params.html" %\
                             (BOOTSTRAP_SIZE, int(id_str)))
        psd.to_html(param_html_stream)
        print "Wrote %s" % (param_html_stream)


if __name__ == '__main__':
    main()

