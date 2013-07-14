import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'axes.linewidth': 2})
matplotlib.rc('text', usetex=False)
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA

class MultipanelPlot(object):
    def __init__(self, rows, cols, figsize=(6,6)):
        self.F = plt.figure(figsize=figsize)
        self.rows = rows
        self.cols = cols
        self.current_ax_id = 1

    def get_new_ax(self):
        ax = AA.Subplot(self.F, self.rows, self.cols, self.current_ax_id)
        return ax

    def add_ax_to_plot(self, ax):
        self.F.add_subplot(ax)
        self.current_ax_id += 1

    def finalize_plot(self, plot_file_name):
        plt.tight_layout()
        plt.draw()
        plt.savefig(plot_file_name, bbox_inches='tight')
        plt.clf()
        print "Wrote %s" % plot_file_name
