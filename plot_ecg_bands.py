import os
import numpy as np
from math import ceil 
import re


import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator



def get_lb_ub(ecg_samples, percentile = 95, perc_method = 'closest_observation'):
    
    #shape of ecg_xx_samples(batch_size, n_samples, 12)
    ub_percent = np.max([100 - percentile, percentile])
    lb_percent = np.min([100 - percentile, percentile])

    sample_ub = np.percentile(ecg_samples, ub_percent, axis = 0, method = perc_method)
    sample_lb = np.percentile(ecg_samples, lb_percent, axis = 0, method = perc_method)
    sample_avg = np.percentile(ecg_samples, 50, axis = 0, method = perc_method)

    return sample_lb, sample_ub, sample_avg


def plot_ecg_bands(ecg_real_sample, ecg_gen_sample, ecg_real_diagnosis= 'Real', ecg_gen_diagnosis= 'Synth', 
                   bands = True, st_start  = 13, st_end = 21, ecg_sampling_frequency = 100, 
                   show_st = False, show_avg_ecg = False, grid_on = False, save_folder_path = None, add_title = True):
    '''
    Plot several ECGs from samples.
    Either plotting them one by one, bands = False
    Or plotting a band between the max and min values of the real and gen samples provided.
    st_start and st_end are the chosen sample indices for the start and end of the st-segment considered
    '''

    leads=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    #check real-fake ecg shape so that the last dimension is 12 (leads)
    if ecg_real_sample.shape[-1] == 12:
        pass
    elif ecg_real_sample.shape[1] == 12:
        ecg_real_sample = np.transpose(ecg_real_sample, (0, 2, 1))
    else:
        raise ValueError(f"Expected one dimension to have size 12, got {ecg_real_sample.shape}")
    
    if ecg_gen_sample.shape[-1] == 12:
        pass
    elif ecg_gen_sample.shape[1] == 12:
        ecg_gen_sample = np.transpose(ecg_gen_sample, (0, 2, 1))
    else:
        raise ValueError(f"Expected one dimension to have size 12, got {ecg_gen_sample.shape}")

    ecg_sample_length = ecg_real_sample.shape[1]
    time_samples = np.arange(0, ecg_sample_length)/ecg_sampling_frequency # SECONDS assuming its is of shape [batch_size, n_samples, 12] 

    
    #Plot all of the windows from samples
    rows=2
    columns=6
    n_lead =12

    secs=0.41
    display_factor= 2
    row_height= 12 #how many grid should a lead signal have
    figsize=(2*columns*2*0.5 , (2*rows * row_height / 5 * display_factor)*0.5)  # Use columns*N for the fisrt entry to have the dilated rectangles of Nx1 mm

    fig, ax_left = plt.subplots(rows, columns, figsize=figsize, sharex=True, sharey='row')
    fig.subplots_adjust(wspace = 0.08, hspace = 0)
    
    if add_title:
        fig.suptitle(f'REAL ECG {ecg_real_diagnosis} (Blue) and Synth from {ecg_gen_diagnosis} (Red)', fontsize=13, va='bottom', y=0.92)
    
    fig.add_gridspec(rows, columns, wspace=0)
    
    #shape of ecg_xx_samples(BATCH_SIZE, n_samples, 12)
    real_ub, real_lb, real_avg, = get_lb_ub(ecg_real_sample, percentile = 95, perc_method = 'closest_observation')
    synth_ub, synth_lb, synth_avg = get_lb_ub(ecg_gen_sample, percentile = 95, perc_method = 'closest_observation')

    
    for col in range(ax_left.shape[1]):
        for row in range(ax_left.shape[0]):

            ax=ax_left[row, col]
            ind=ax_left.shape[0]*col+row
            
            if bands:
                #Real, band and plot
                ax.fill_between(time_samples, real_lb[:, ind], real_ub[:, ind], 
                                label=f'Real ECG {ecg_real_diagnosis}, {len(ecg_real_sample)} samples', color ='b', alpha = 0.7) 
                if show_avg_ecg:
                    ax.plot(time_samples, real_avg[:,ind], label=f'Real ECG {ecg_real_diagnosis}, average', color='b')

                #gen band and plot
                ax.fill_between(time_samples, synth_lb[:, ind], synth_ub[:, ind], 
                                label=f'SYNTH ECG from {ecg_gen_diagnosis}, {len(ecg_gen_sample)} samples', color ='r', alpha = 0.7) #color ='m'
                if show_avg_ecg:
                    ax.plot(time_samples, synth_avg[:,ind], linewidth=2, label=f'SYNTH from {ecg_gen_diagnosis} ECG', color='r')
            
            else: #Plot MANY ECGS TOGHETHER
                for ir in range(len(ecg_real_sample)):
                    ax.plot(time_samples, ecg_real_sample[ir, :, ind], label=f'Real ECG {ecg_real_diagnosis}', color ='b') #color ='b'
                for ig in range(len(ecg_gen_sample)):
                    ax.plot(time_samples, ecg_gen_sample[ig, :, ind], label=f'SYNTH ECG from {ecg_gen_diagnosis}', color ='r') #color ='m'

            if show_st:
                start_sec = time_samples[st_start]
                end_sec = time_samples[st_end]
                ax.vlines([start_sec, end_sec], -3, 3, linestyle='dashed', color = 'g', linewidth=1.5)
                ax.vlines(st_end, -3, 3, linestyle='dashed', color = 'k', linewidth='1')

            lead_name=leads[ind]
            ax.set_title(f'{lead_name}', loc='center', fontsize = 8)
            ax.tick_params('y', colors='#3979f0')
            
            xtick=np.arange(0,ecg_sample_length +1,20)/ecg_sampling_frequency

            ax.set_xticks(xtick)  
            ax.set_yticks(np.arange(-3,3,0.5))
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            ax.minorticks_on()
            
            ax.xaxis.set_minor_locator(AutoMinorLocator(5)) # Set to 5 for reactangles of 2x1 instead of 1x1
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))

            ax.set_ylim(-3, 3)
            ax.set_xlim(0, time_samples[-1]+1/ecg_sampling_frequency)

            if grid_on:
                ax.grid(which='major', linestyle='-', linewidth='0.5', color='#A9A9A9')
                ax.grid(which='minor', linestyle='-', linewidth='0.5', color='#D3D3D3')
        
    lines = []
    labels = []
    for ax in fig.axes:
        Line, Label = ax.get_legend_handles_labels()
        # print(Label)
        lines.extend(Line)
        labels.extend(Label)
        
    fig.text(0.5, 0.04, '(s)', ha='center')  # X-axis label
    fig.text(0.04, 0.5, '(mV)', va='center', rotation='vertical')  # Y-axis label
    
        
    if save_folder_path is not None: 
        if save_folder_path.endswith('.pdf'): #If I give the full path with the filename, I will use it as is
            additional_part = f"_{ecg_real_diagnosis}_VS_{ecg_gen_diagnosis}.pdf"
            figtitle=re.sub(r'.pdf', additional_part, save_folder_path)
            
        else:
            if bands:
                figtitle = os.path.join(save_folder_path, f'real_{ecg_real_diagnosis}_VS_synthFrom_{ecg_gen_diagnosis}_ecg_BANDS_{len(ecg_gen_sample)}elem.pdf')
            else:
                figtitle = os.path.join(save_folder_path, f'real_{ecg_real_diagnosis}_VS_synthFrom_{ecg_gen_diagnosis}_ecg_SAMPLES_{len(ecg_gen_sample)}elem.pdf')

        plt.rc('pdf', fonttype = 42)
        plt.rc('ps', fonttype = 42)
        plt.savefig(figtitle, transparent=False, bbox_inches='tight', pad_inches=0, rasterized=True)
        plt.close(fig)
    
    else:
        plt.show()
        
    return



def plot_bands_datasets(ecg_dataset, save_figs_folder = None, bands = True, ecg_sampling_frequency = 100, show_avg_ecg = False, grid_on = False):
    
    '''
    Takes as input the dataset created in utils make_data_class.
    '''
    
    if len(ecg_dataset) == 0:
        raise ValueError("The list of ECG data pairs is empty.")
    
    for pair in ecg_dataset:
        plot_ecg_bands(pair.real.data, pair.fake.data, pair.real.label, pair.fake.label,
                       bands = bands,  ecg_sampling_frequency = ecg_sampling_frequency,
                       show_avg_ecg = show_avg_ecg, grid_on = grid_on, save_folder_path=save_figs_folder)
        
        
    return