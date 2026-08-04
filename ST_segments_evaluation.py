import numpy as np
import scipy
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = False

from radar_chart import *

import os


def compute_avg_percent(real_avg, fake_avg, signif = 0.05):

    real_mu = np.average(real_avg, axis=0)
    fake_mu = np.average(fake_avg, axis=0)

    mu_diff = real_mu - fake_mu

    real_cov = np.cov(real_avg.T)
    fake_cov = np.cov(fake_avg.T)

    n_real = real_avg.shape[0]
    n_fake = fake_avg.shape[0]

    dof = len(real_mu)

    chi2_dof_sig = scipy.stats.chi2.ppf(1-signif, dof)

    cov_sum = np.array([real_cov[i,i]*(1/n_real)+fake_cov[i,i]*(1/n_fake) for i in range(real_cov.shape[0])])

    conf_int_lb = mu_diff - np.sqrt(chi2_dof_sig*cov_sum)
    conf_int_ub = mu_diff + np.sqrt(chi2_dof_sig*cov_sum)
    
    return mu_diff, conf_int_lb, conf_int_ub


def make_data_ecg(diff_avg, ci_lb, ci_ub):
    data_diff = np.array([ci_lb, diff_avg, ci_ub])
    data = [ ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5','V6'],
    ('data', data_diff)]

    return data

def plot_st_radar(real_data, fake_data, st_start, st_end, 
                  real_label="Real", fake_label="Fake", significance=0.05, fig_size = (7,7),
                  title = None, show_title = True, show_plot = False, ax = None, save_folder = None):
    
    """
    Plot one ST-segment radar comparison.

    Parameters
    ----------
    real_data : np.ndarray
        Shape: (n_real_ecgs, n_timepoints, 12)
    fake_data : np.ndarray
        Shape: (n_fake_ecgs, n_timepoints, 12)
    st_start : int
        Inclusive ST-segment start index.
    st_end : int
        Exclusive ST-segment end index.
    """
    
    real_st_avg = np.mean( real_data[:, st_start:st_end, :], axis=1, )
    fake_st_avg = np.mean( fake_data[:, st_start:st_end, :], axis=1, )

    diff_avg, conf_int_lb, conf_int_ub = compute_avg_percent(real_st_avg, fake_st_avg, significance)
    
    # plt.rcParams['text.usetex'] = True

    N = 12
    theta = radar_factory(N, frame='polygon')
    
    data = make_data_ecg(diff_avg, conf_int_lb, conf_int_ub)
    spoke_labels = data.pop(0)

    if ax==None:
        fig, ax = plt.subplots(figsize=fig_size, nrows=1, ncols=1,
                                    subplot_kw=dict(projection='radar'))
        
        if show_title:
            if title == None:
                title = r'Confidence bounds $ (\overline{x}^{R}_{lead} - \overline{y}^{S}_{lead}) \pm \sqrt{ \chi^2_{12, \alpha} (S^{R}_{lead}/|X_{R}| + S^{S}_{lead}/|Y_{S}|)} $, where $ \overline{x}^{R}_{lead}$ is ' + real_label + r', and $ \overline{y}^{S}_{lead} $ '+ fake_label
        fig.text(0.5, 0.965, title, horizontalalignment='center', color='black', weight='bold', size='large')
    #fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'b']

    if show_title:
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')
    
    for ind, color in enumerate(colors):
        ax.plot(theta, data[0][1][ind, :], color=color)  
    ax.set_varlabels(spoke_labels)

    ax.plot(theta, [0]*12, color='k', linestyle='--')


    data_fb_0 = data[0][1][0, :]
    data_fb_2 = data[0][1][2, :]
    ax.fill_between(theta, data_fb_0, data_fb_2, facecolor='b', alpha=0.10)
    ax.set_rgrids(np.linspace(-0.125, 0.125, num=5))
    ax.set_title(f'{real_label} and {fake_label}', weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')

    # add legend relative to top-left plot
    labels = ('Lowerbound', 'Average', 'Upper bound')
    legend = ax.legend(labels, loc=(0.9, .95),
                                labelspacing=0.1, fontsize='small')

    if save_folder != None:
        save_name = os.path.join(save_folder, f'ST_RadarPlot_{real_label}_{fake_label}.pdf')
        print(f'Saved plot at {save_name}')
    
    if show_plot:
        plt.show()
    
    return ax


def plot_multiple_st_radar(datasets, st_start =35, st_end=50,
                        significance=0.05, fig_size=(20,20), 
                        show_title=True, show_plot=False,
                        save_folder=None, save_singles=False, show_title_single = False):
    
    #Compute ideal number of grid rows and columns for subplots
    n_plots = len(datasets)
    if n_plots == 1:
        n_rows = 1
        n_cols = 1
    
    else:
        #Compute Grid Shape
        possible_grids = []
        for n_cols in range(1, min(5, n_plots) + 1):
            n_rows = int(np.ceil(n_plots / n_cols))
            empty_axes = n_rows * n_cols - n_plots
            grid_difference = abs(n_rows - n_cols)
            
            possible_grids.append( (empty_axes, grid_difference, n_rows, n_cols) )
        _, _, n_rows, n_cols = min(possible_grids)
    
    radar_factory(12, frame='polygon')
    fig, axs = plt.subplots( figsize=fig_size, nrows=n_rows, ncols=n_cols, subplot_kw=dict(projection='radar') )
    axs = np.asarray(axs).reshape(n_rows, n_cols)

    
    for ind, pair in enumerate(datasets):
        
        row = ind // n_cols
        col = ind % n_cols
        ax = axs[row, col]
        
        real_data = np.asarray(pair.real.data)
        fake_data = np.asarray(pair.fake.data)
        
        # Convert (batch_size, 12, samples)
        # to      (batch_size, samples, 12)
        if real_data.shape[-1] != 12:
            real_data = np.transpose(real_data, (0, 2, 1))

        if fake_data.shape[-1] != 12:
            fake_data = np.transpose(fake_data, (0, 2, 1))

        if save_singles:
            save_single_folder = save_folder
            show_title_single = True
        else:
            save_single_folder = None
        
        ax = plot_st_radar( real_data=real_data, fake_data=fake_data, st_start=st_start, st_end=st_end, 
                           real_label=pair.real.label, fake_label=pair.fake.label, significance=significance,
                            title=None, show_title=show_title_single, show_plot=False, ax=ax, save_folder=save_single_folder)
        
        
        ax.set_title( f'{pair.real.label} and {pair.fake.label}', weight='bold', size='medium', 
                     position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center' )
        
        
            # Remove empty radar plots
    for ind in range(len(datasets), n_rows * n_cols):
        row = ind // n_cols
        col = ind % n_cols
        axs[row, col].set_visible(False)

    if show_title:
        title='ST-segment comparison'
        fig.suptitle(title, weight='bold', size='large')

    if save_folder is not None:
        save_name = os.path.join(save_folder, 'ST_RadarPlot_ALL.pdf')
        fig.savefig(save_name, bbox_inches='tight')

    if show_plot:
        plt.show()
    
    return 


