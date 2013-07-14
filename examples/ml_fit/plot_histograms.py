import sys
import os.path
import numpy
from multipanel import MultipanelPlot
from palm.blink_target_data import BlinkCollectionTargetData
from palm.parameter_set_distribution import ParamSetDistFactory,\
                                            ParameterSetDistribution
from palm.blink_parameter_set import SingleDarkParameterSet

DATA = os.path.expanduser("./params")
COLOR_SCHEME = ["#EAB086", "#EAC786", "#7995C6", "#6FC2B9", '0.65']
LIMIT = 1.5
DELTA = 1.0

def load_psd(filename):
    psd = ParameterSetDistribution()
    psd.load_from_file(filename)
    return psd

def plot_one(ax, data, true_val, xaxis_label, yaxis_label, num_bins,
             x_lim, y_lim, x_ticks, y_ticks, color='k'):
    bins = numpy.linspace(x_lim[0], x_lim[1], num_bins)
    ax.hist( data, bins=bins, normed=False, color=color,
             edgecolor='k', lw=0.3, alpha=0.8)
    ax.plot([true_val, true_val], [0, y_lim[1]], 'k--', lw=1)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    x_ticklabels = [str(x) for x in x_ticks]
    y_ticklabels = [str(y) for y in y_ticks]
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])

def plot_N_hist(N_array, plot_name):
    mean_estimate = numpy.mean(N_array)
    std_estimate = numpy.std(N_array)
    alpha = 5.0 # percent
    lower_bound = alpha * 0.5
    upper_bound = 100 - (alpha * 0.5)
    CI = numpy.percentile(N_array, [lower_bound, upper_bound])
    print "N", mean_estimate, std_estimate, CI
    xaxis_label = r"$\mathbf{N}$"
    yaxis_label = r"$\mathbf{counts}$"
    bins = numpy.arange(0.5, 5.5, 1.0)
    x_lim = (0, 5)
    x_ticks = range(0, 6)
    y_lim = (0, 300)
    y_ticks = [100, 200, 300]
    true_val = 1

    mp = MultipanelPlot(1, 1, figsize=(6,6))
    ax = mp.get_new_ax()
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.hist( N_array, bins=bins, normed=False, color=COLOR_SCHEME[-1],
             edgecolor='k', lw=0.3, alpha=0.8)
    ax.plot([true_val, true_val], [0, y_lim[1]], 'k--', lw=1)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    x_ticklabels = [str(x) for x in x_ticks]
    y_ticklabels = [str(y) for y in y_ticks]
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    mp.add_ax_to_plot(ax)
    mp.finalize_plot(plot_name)

def plot_rates_hist(ka_array, kb_array, kd_array, kr_array, plot_name):
    data = [(ka_array, -0.3, r"$\mathbf{log(k_a)}$", r"$\mathbf{counts}$", 20, 0),
            (kb_array, 0.00, r"$\mathbf{log(k_b)}$", "", 20, 3),
            (kd_array, 0.48, r"$\mathbf{log(k_d)}$", r"$\mathbf{counts}$", 20, 1),
            (kr_array, -1.0, r"$\mathbf{log(k_r)}$", "", 20, 2) ]
    mp = MultipanelPlot(2, 2, figsize=(10,10))
    for i, p in enumerate(data):
        this_data_array = p[0]
        true_val = p[1]
        this_xaxis_label = p[2]
        this_yaxis_label = p[3]
        num_bins = p[4]
        color_id = p[5]
        print this_data_array.shape
        mean_estimate = numpy.mean(this_data_array)
        std_estimate = numpy.std(this_data_array)
        alpha = 5.0 # percent
        lower_bound = alpha * 0.5
        upper_bound = 100 - (alpha * 0.5)
        CI = numpy.percentile(this_data_array, [lower_bound, upper_bound])
        print this_xaxis_label, mean_estimate, std_estimate, CI
        print "%s, %.2e, %s" %\
              (this_xaxis_label, 10**mean_estimate, 10**numpy.array(CI))
        x_range = (this_data_array.min(), this_data_array.max())
        x_lim = (true_val-LIMIT, true_val+LIMIT)
        # x_lim = (mean_estimate-LIMIT, mean_estimate+LIMIT)
        y_lim = (0, 100)
        x_ticks = numpy.arange(true_val-LIMIT, true_val+LIMIT+DELTA, DELTA)
        # x_ticks = numpy.arange(mean_estimate-LIMIT, mean_estimate+LIMIT+DELTA,
                               # DELTA)
        y_ticks = numpy.arange(20, 120, 20)
        ax = mp.get_new_ax()
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)
        plot_one(ax, this_data_array, true_val,
                 xaxis_label=this_xaxis_label,
                 yaxis_label=this_yaxis_label,
                 num_bins=num_bins, x_lim=x_lim, y_lim=y_lim, x_ticks=x_ticks,
                 y_ticks=y_ticks, color=COLOR_SCHEME[color_id])
        mp.add_ax_to_plot(ax)
    mp.finalize_plot(plot_name)

def main():
    try:
        bootstrap_size = int(sys.argv[1])
    except IndexError:
        print "\nExample usage:\npython plot_histograms.py <bootstrap_size>\n" \
              "\twhere <bootstrap_size> is an integer that corresponds to\n" \
              "\tthe value of BOOTSTRAP_SIZE in bootstrap_ml_fit.py\n"
        raise
    base_name = "%03d" % bootstrap_size
    p = os.path.join(DATA, "best.pkl")
    psd = load_psd(p)

    ka_array = psd.single_parameter_distribution_as_array('log_ka')
    kd_array = psd.single_parameter_distribution_as_array('log_kd')
    kr_array = psd.single_parameter_distribution_as_array('log_kr')
    kb_array = psd.single_parameter_distribution_as_array('log_kb')
    N_array = psd.single_parameter_distribution_as_array('N')

    rate_plot_name = os.path.join('histograms', base_name + '_rate_hist.pdf')
    plot_rates_hist(ka_array, kb_array, kd_array, kr_array, rate_plot_name)
    
    N_plot_name = os.path.join('histograms', base_name + '_N_hist.pdf')
    plot_N_hist(N_array, N_plot_name)

if __name__ == '__main__':
    main()

