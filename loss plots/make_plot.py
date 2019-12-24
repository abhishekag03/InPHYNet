import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# import subprocess
# subprocess.check_call(["latex"])
# import os 
# os.environ["PATH"] += os.pathsep + '/usr/bin'

# aux_loss_history = pickle.load(open('inphynet_aux_loss_tfidf.pickle', 'rb'))
prim_loss_history = pickle.load(open('inphynet_prim_loss_tfidf.pickle', 'rb'))
total_loss_history = pickle.load(open('inphynet_total_loss_tfidf.pickle', 'rb'))
n_epochs = 50


SPINE_COLOR = 'gray'
plt.style.use(['seaborn'])
#use 'bmh' for grey plots
width = 345
	
plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title

csfont = {'fontname':'serif'}
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 13,
        "font.size": 13,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
}

mpl.rcParams.update(nice_fonts)

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def plot_required(history, filename, y_label):
    n_epochs = len(history)+1

    fig, ax1 = plt.subplots(1, 1, figsize=set_size(width))
    # ax1=fig.add_subplot(1, 3, 1)
    ax1.plot(np.arange(1, n_epochs), history)
    ax1.set_xlabel('Epochs', **csfont)
    ax1.set_ylabel(y_label, **csfont)
    plt.savefig('./plots/'+filename+".pdf", format='pdf', bbox_inches='tight')
    # plt.show()

plot_required(prim_loss_history, "prim_loss_history_tfidf_inphynet", 'Primary Task Training Loss')
plot_required(total_loss_history, "total_loss_history_tfidf_inphynet", 'Total Training Loss')

# plt.figure(1)
# plt.plot(np.arange(1, len(aux_loss_history)+1), aux_loss_history)
# plt.xlabel('Epochs')
# plt.ylabel('Auxillary Task Training Loss')
# plt.grid(True)
# plt.savefig('aux_loss.png')
# plt.show()

# plt.figure(2)
# plt.plot(np.arange(1, len(prim_loss_history)+1), prim_loss_history)
# plt.xlabel('Epochs')
# plt.ylabel('Primary Task Training Loss')
# plt.grid(True)

# plt.savefig('prim_loss.png')

# plt.show()

# plt.figure(3)
# plt.plot(np.arange(1, len(total_loss_history)+1), total_loss_history)
# plt.xlabel('Epochs')
# plt.ylabel('Total Training Loss')
# plt.grid(True)
# plt.savefig('total_loss.png')

# plt.show()

print("Done")