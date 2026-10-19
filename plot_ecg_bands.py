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
                ax.fill_between(time_samples, real_lb[:, ind], real_ub[:, ind], rasterized=True,
                                label=f'Real ECG {ecg_real_diagnosis}, {len(ecg_real_sample)} samples', color ='b', alpha = 0.7) 
                if show_avg_ecg:
                    ax.plot(time_samples, real_avg[:,ind], label=f'Real ECG {ecg_real_diagnosis}, average', color='b', rasterized=True)

                #gen band and plot
                ax.fill_between(time_samples, synth_lb[:, ind], synth_ub[:, ind], rasterized=True,
                                label=f'SYNTH ECG from {ecg_gen_diagnosis}, {len(ecg_gen_sample)} samples', color ='r', alpha = 0.7) #color ='m'
                if show_avg_ecg:
                    ax.plot(time_samples, synth_avg[:,ind], linewidth=2, label=f'SYNTH from {ecg_gen_diagnosis} ECG', color='r', rasterized=True)
            
            else: #Plot MANY ECGS TOGHETER
                for ir in range(len(ecg_real_sample)):
                    ax.plot(time_samples, ecg_real_sample[ir, :, ind], label=f'Real ECG {ecg_real_diagnosis}', color ='b', rasterized=True) #color ='b'
                for ig in range(len(ecg_gen_sample)):
                    ax.plot(time_samples, ecg_gen_sample[ig, :, ind], label=f'SYNTH ECG from {ecg_gen_diagnosis}', color ='r', rasterized=True) #color ='m'

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
        plt.savefig(figtitle, transparent=False, bbox_inches=None, pad_inches=0)
        plt.close(fig)
    
    else:
        plt.show()
        
    return



def plot_bands_datasets(ecg_dataset, save_figs_folder = None, bands = True, ecg_sampling_frequency = 100, show_avg_ecg = False, grid_on = False, remove_baseline = False):
    
    '''
    Takes as input the dataset created in utils make_data_class.
    If remove_baseline is True, the baseline is removed independently from
    every heartbeat and lead before computing and plotting the bands.
    '''
    
    if len(ecg_dataset) == 0:
        raise ValueError("The list of ECG data pairs is empty.")
    
    for pair in ecg_dataset:
        ecg_real_sample = np.asarray(pair.real.data).copy()
        ecg_gen_sample = np.asarray(pair.fake.data).copy()

        if remove_baseline:
            if ecg_real_sample.ndim != 3 or ecg_gen_sample.ndim != 3:
                raise ValueError("Real and generated ECG data must be 3-D arrays.")

            if ecg_real_sample.shape[-1] == 12:
                ecg_real_sample = np.transpose(ecg_real_sample, (0, 2, 1))
            elif ecg_real_sample.shape[1] != 12:
                raise ValueError(f"Expected one real ECG dimension to have size 12, got {ecg_real_sample.shape}")

            if ecg_gen_sample.shape[-1] == 12:
                ecg_gen_sample = np.transpose(ecg_gen_sample, (0, 2, 1))
            elif ecg_gen_sample.shape[1] != 12:
                raise ValueError(f"Expected one generated ECG dimension to have size 12, got {ecg_gen_sample.shape}")

            if ecg_real_sample.shape[-1] < 100 or ecg_gen_sample.shape[-1] < 100:
                raise ValueError("Baseline removal requires ECGs with at least 100 time samples.")

            real_baseline = find_baseline(10, 100, 2, 20, ecg_real_sample)
            gen_baseline = find_baseline(10, 100, 2, 20, ecg_gen_sample)
            ecg_real_sample = ecg_real_sample - real_baseline[:, :, None]
            ecg_gen_sample = ecg_gen_sample - gen_baseline[:, :, None]

        plot_ecg_bands(ecg_real_sample, ecg_gen_sample, pair.real.label, pair.fake.label, bands = bands, ecg_sampling_frequency = ecg_sampling_frequency, show_avg_ecg = show_avg_ecg, grid_on = grid_on, save_folder_path = save_figs_folder)
        
        
    return




def find_baseline( start_sample_index, end_sample_index, stride, window_length, ecg, ):
    """
    ecg: (12, samples) or (batch, 12, samples)
    
    What this does:
    Finds the baseline of an ECG signal by identifying the window with the least change in amplitude.
    
    - Selects windows of window length
    - Run this window across the ECG signal from start_sample_index to end_sample_index with a given stride
    - For each window, calculate the mean absolute difference between consecutive samples (i.e., the change in amplitude)
    - Select the window with the least change in amplitude AMONG ALL LEADS as the baseline
    - The baseline is then calculated as the median of the best window across all leads.
    
    Output shape: (12,) or (batch, 12) baseline values for each lead in the ECG signal.
    
    """

    single_ecg = ecg.ndim == 2

    if single_ecg:
        ecg = ecg[None, :, :]

    window_starts = range(start_sample_index, end_sample_index - window_length + 1, stride, )
    window_changes = []

    for start in window_starts:
        window = ecg[:, :, start:start + window_length]

        change = np.abs(np.diff(window, axis=-1)).mean(axis=(1, 2))

        window_changes.append(change)

    window_changes = np.stack(window_changes, axis=1)

    best_window_indices = np.argmin(window_changes, axis=1)

    baselines = []

    for batch_index, window_index in enumerate(best_window_indices):
        start = window_starts[window_index]
        best_window = ecg[ batch_index, :, start:start + window_length, ]

        baselines.append(np.median(best_window, axis=-1))

    baselines = np.stack(baselines)

    if single_ecg:
        return baselines[0]

    return baselines



# def plot_bands_datasets_IoU(ecg_dataset, save_figs_folder = None, bands = True, 
#                             ecg_sampling_frequency = 100, show_avg_ecg = False, grid_on = False,
#                             show_plots = True):
    
#     '''
#     Takes as input the dataset created in utils make_data_class.
#     Returns (real_label, fake_label, IoU) for each pair.
#     '''
    
#     if len(ecg_dataset) == 0:
#         raise ValueError("The list of ECG data pairs is empty.")
    
#     leads=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#     results = []
    
#     for pair in ecg_dataset:
        
#         ecg_real_sample = pair.real.data
#         ecg_gen_sample = pair.fake.data
#         ecg_real_diagnosis = pair.real.label
#         ecg_gen_diagnosis = pair.fake.label
        
#         #check real-fake ecg shape so that the last dimension is 12 (leads)
#         if ecg_real_sample.shape[-1] == 12:
#             pass
#         elif ecg_real_sample.shape[1] == 12:
#             ecg_real_sample = np.transpose(ecg_real_sample, (0, 2, 1))
#         else:
#             raise ValueError(f"Expected one dimension to have size 12, got {ecg_real_sample.shape}")
        
#         if ecg_gen_sample.shape[-1] == 12:
#             pass
#         elif ecg_gen_sample.shape[1] == 12:
#             ecg_gen_sample = np.transpose(ecg_gen_sample, (0, 2, 1))
#         else:
#             raise ValueError(f"Expected one dimension to have size 12, got {ecg_gen_sample.shape}")
        
#         if ecg_real_sample.shape[1:] != ecg_gen_sample.shape[1:]:
#             raise ValueError("Real and fake ECGs must have the same number of time samples and leads.")
        
#         #90% bands, between the 5th and 95th percentiles
#         real_lb, real_ub, real_avg = get_lb_ub(ecg_real_sample, percentile = 95, perc_method = 'closest_observation')
#         synth_lb, synth_ub, synth_avg = get_lb_ub(ecg_gen_sample, percentile = 95, perc_method = 'closest_observation')
        
#         intersection = np.maximum(0, np.minimum(real_ub, synth_ub) - np.maximum(real_lb, synth_lb))
#         union = (real_ub - real_lb) + (synth_ub - synth_lb) - intersection
        
#         #sum across time samples and leads; undefined if both bands have zero area
#         union_sum = np.sum(union)
#         IoU = float(np.sum(intersection) / union_sum) if union_sum > 0 else np.nan
#         results.append((ecg_real_diagnosis, ecg_gen_diagnosis, IoU))
        
#         ecg_sample_length = ecg_real_sample.shape[1]
#         time_samples = np.arange(0, ecg_sample_length)/ecg_sampling_frequency
        
#         rows=2
#         columns=6
#         display_factor=2
#         row_height=12
#         figsize=(2*columns*2*0.5, (2*rows * row_height / 5 * display_factor)*0.5)
        
#         fig, ax_left = plt.subplots(rows, columns, figsize=figsize, sharex=True, sharey='row')
#         fig.subplots_adjust(wspace = 0.08, hspace = 0)
#         fig.suptitle(f'REAL ECG {ecg_real_diagnosis} (Blue) and Synth from {ecg_gen_diagnosis} (Red), IoU = {IoU:.4f}',
#                      fontsize=13, va='bottom', y=0.92)
        
#         for col in range(ax_left.shape[1]):
#             for row in range(ax_left.shape[0]):
                
#                 ax=ax_left[row, col]
#                 ind=ax_left.shape[0]*col+row
                
#                 if bands:
#                     ax.fill_between(time_samples, real_lb[:, ind], real_ub[:, ind],
#                                     rasterized=True, color='b', alpha = 0.7)
#                     ax.fill_between(time_samples, synth_lb[:, ind], synth_ub[:, ind],
#                                     rasterized=True, color='r', alpha = 0.7)
                    
#                     if show_avg_ecg:
#                         ax.plot(time_samples, real_avg[:, ind], color='b', rasterized=True)
#                         ax.plot(time_samples, synth_avg[:, ind], linewidth=2, color='r', rasterized=True)
                
#                 else:
#                     for ir in range(len(ecg_real_sample)):
#                         ax.plot(time_samples, ecg_real_sample[ir, :, ind], color='b', rasterized=True)
#                     for ig in range(len(ecg_gen_sample)):
#                         ax.plot(time_samples, ecg_gen_sample[ig, :, ind], color='r', rasterized=True)
                
#                 ax.set_title(f'{leads[ind]}', loc='center', fontsize = 8)
#                 ax.tick_params('y', colors='#3979f0')
                
#                 xtick=np.arange(0, ecg_sample_length +1, 20)/ecg_sampling_frequency
#                 ax.set_xticks(xtick)
#                 ax.set_yticks(np.arange(-3, 3, 0.5))
#                 ax.tick_params(axis='x', labelsize=7)
#                 ax.tick_params(axis='y', labelsize=7)
                
#                 ax.minorticks_on()
#                 ax.xaxis.set_minor_locator(AutoMinorLocator(5))
#                 ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                
#                 ax.set_ylim(-3, 3)
#                 ax.set_xlim(0, time_samples[-1]+1/ecg_sampling_frequency)
                
#                 if grid_on:
#                     ax.grid(which='major', linestyle='-', linewidth='0.5', color='#A9A9A9')
#                     ax.grid(which='minor', linestyle='-', linewidth='0.5', color='#D3D3D3')
        
#         fig.text(0.5, 0.04, '(s)', ha='center')
#         fig.text(0.04, 0.5, '(mV)', va='center', rotation='vertical')
        
#         if save_figs_folder is not None:
#             if save_figs_folder.endswith('.pdf'):
#                 additional_part = f"_{ecg_real_diagnosis}_VS_{ecg_gen_diagnosis}_IoU.pdf"
#                 figtitle=re.sub(r'\.pdf$', additional_part, save_figs_folder)
#             else:
#                 if bands:
#                     figtitle = os.path.join(save_figs_folder, f'real_{ecg_real_diagnosis}_VS_synthFrom_{ecg_gen_diagnosis}_ecg_BANDS_IoU_{len(ecg_gen_sample)}elem.pdf')
#                 else:
#                     figtitle = os.path.join(save_figs_folder, f'real_{ecg_real_diagnosis}_VS_synthFrom_{ecg_gen_diagnosis}_ecg_SAMPLES_IoU_{len(ecg_gen_sample)}elem.pdf')
            
#             plt.rc('pdf', fonttype = 42)
#             plt.rc('ps', fonttype = 42)
#             fig.savefig(figtitle, transparent=False, bbox_inches=None, pad_inches=0)
#             plt.close(fig)
        
#         else:
#             if show_plots:
#                 plt.show()
    
#     return results





def plot_bands_datasets_IoU( ecg_dataset, bands_percentile = 95, save_figs_folder=None, bands=True, ecg_sampling_frequency=100,
    show_avg_ecg=False, grid_on=False, show_plots=True, remove_baseline=False, ):
    """
    Plot real and generated ECG distributions and calculate their band IoU.

    If ``remove_baseline=True``, a baseline is estimated independently for
    every heartbeat and lead using ``find_baseline``. The baseline-corrected
    ECGs are then used to calculate the percentile bands, IoU, averages,
    and plots.

    Parameters
    ----------
    ecg_dataset
        Dataset created by ``make_data_class``.

    remove_baseline : bool, default=False
        Whether to remove the per-heartbeat, per-lead baseline before
        calculating the bands and IoU.

    Returns
    -------
    results : list of tuples
        Each tuple contains:
        ``(real_label, fake_label, IoU)``.
    """

    if len(ecg_dataset) == 0:
        raise ValueError("The list of ECG data pairs is empty.")

    leads = [ "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6", ]

    # Same baseline-search configuration used in
    # a00_further_ecg_eval_metrics.py.
    bl_start_sample_index = 10
    bl_end_sample_index = 100
    bl_stride = 2
    bl_window_length = 20

    results = []

    for pair in ecg_dataset:

        # Copy the data so baseline removal does not modify the dataset.
        ecg_real_sample = np.asarray(pair.real.data).copy()
        ecg_gen_sample = np.asarray(pair.fake.data).copy()

        ecg_real_diagnosis = pair.real.label
        ecg_gen_diagnosis = pair.fake.label

        # Convert both arrays to:
        # (number_of_heartbeats, time_samples, 12_leads)
        if ecg_real_sample.ndim != 3:
            raise ValueError( "Real ECG data must be a 3-D array, but its shape is " f"{ecg_real_sample.shape}." )

        if ecg_real_sample.shape[-1] == 12:
            pass
        elif ecg_real_sample.shape[1] == 12:
            ecg_real_sample = np.transpose(ecg_real_sample, (0, 2, 1))
        else:
            raise ValueError( "Expected one real ECG dimension to have size 12, " f"but got {ecg_real_sample.shape}." )

        if ecg_gen_sample.ndim != 3:
            raise ValueError( "Generated ECG data must be a 3-D array, but its shape is " f"{ecg_gen_sample.shape}." )

        if ecg_gen_sample.shape[-1] == 12:
            pass
        elif ecg_gen_sample.shape[1] == 12:
            ecg_gen_sample = np.transpose(ecg_gen_sample, (0, 2, 1))
        else:
            raise ValueError( "Expected one generated ECG dimension to have size 12, " f"but got {ecg_gen_sample.shape}." )

        if ecg_real_sample.shape[1:] != ecg_gen_sample.shape[1:]:
            raise ValueError( "Real and generated ECGs must have the same number " "of time samples and leads." )

        if remove_baseline:
            ecg_sample_length = ecg_real_sample.shape[1]

            if ecg_sample_length < bl_end_sample_index:
                raise ValueError( "Baseline removal requires at least " f"{bl_end_sample_index} time samples, but the ECGs " f"contain {ecg_sample_length}." )

            # find_baseline expects:
            # (batch, 12 leads, time samples)
            real_leads_first = np.transpose(ecg_real_sample, (0, 2, 1))
            gen_leads_first = np.transpose(ecg_gen_sample, (0, 2, 1))

            # Returned shape:
            # (number_of_heartbeats, 12)
            real_baseline = find_baseline( bl_start_sample_index, bl_end_sample_index, bl_stride, bl_window_length, real_leads_first, )

            gen_baseline = find_baseline( bl_start_sample_index, bl_end_sample_index, bl_stride, bl_window_length, gen_leads_first, )

            # ecg_*_sample shape: (batch, time, leads)
            # baseline shape after adding an axis: (batch, 1, leads)
            ecg_real_sample = ( ecg_real_sample - real_baseline[:, np.newaxis, :] )
            ecg_gen_sample = ( ecg_gen_sample - gen_baseline[:, np.newaxis, :] )
        
        # 90% bands between the 5th and 95th percentiles.
        real_lb, real_ub, real_avg = get_lb_ub( ecg_real_sample, percentile=bands_percentile, perc_method="closest_observation", )

        synth_lb, synth_ub, synth_avg = get_lb_ub( ecg_gen_sample, percentile=bands_percentile, perc_method="closest_observation", )

        intersection = np.maximum( 0, np.minimum(real_ub, synth_ub) - np.maximum(real_lb, synth_lb), )

        union = ( (real_ub - real_lb) + (synth_ub - synth_lb) - intersection )

        # Sum across time samples and leads.
        # IoU is undefined when both bands have zero total area.
        union_sum = np.sum(union)

        IoU = ( float(np.sum(intersection) / union_sum) if union_sum > 0 else np.nan )
        results.append( (ecg_real_diagnosis, ecg_gen_diagnosis, IoU) )

        ecg_sample_length = ecg_real_sample.shape[1]

        time_samples = ( np.arange(ecg_sample_length) / ecg_sampling_frequency )

        rows = 2
        columns = 6
        display_factor = 2
        row_height = 12

        figsize = ( 2 * columns * 2 * 0.5, (2 * rows * row_height / 5 * display_factor) * 0.5, )

        fig, ax_left = plt.subplots( rows, columns, figsize=figsize, sharex=True, sharey="row", )

        fig.subplots_adjust(wspace=0.08, hspace=0)

        baseline_text = ( ", baseline removed" if remove_baseline else "" )

        fig.suptitle(
            f"REAL ECG {ecg_real_diagnosis} (Blue) and "
            f"Synth from {ecg_gen_diagnosis} (Red), "
            f"IoU = {IoU:.4f}{baseline_text}",
            fontsize=13,
            va="bottom",
            y=0.92,
        )

        for col in range(ax_left.shape[1]):
            for row in range(ax_left.shape[0]):

                ax = ax_left[row, col]
                ind = ax_left.shape[0] * col + row

                if bands:
                    ax.fill_between( time_samples, real_lb[:, ind], real_ub[:, ind], rasterized=True, color="b", alpha=0.7, )

                    ax.fill_between( time_samples, synth_lb[:, ind], synth_ub[:, ind], rasterized=True, color="r", alpha=0.7, )

                    if show_avg_ecg:
                        ax.plot( time_samples, real_avg[:, ind], color="b", rasterized=True, )

                        ax.plot( time_samples, synth_avg[:, ind], linewidth=2, color="r", rasterized=True, )

                else:
                    for real_ecg in ecg_real_sample:
                        ax.plot( time_samples, real_ecg[:, ind], color="b", rasterized=True, )

                    for gen_ecg in ecg_gen_sample:
                        ax.plot( time_samples, gen_ecg[:, ind], color="r", rasterized=True, )

                ax.set_title( leads[ind], loc="center", fontsize=8, )

                ax.tick_params("y", colors="#3979f0")

                xtick = ( np.arange(0, ecg_sample_length + 1, 20) / ecg_sampling_frequency )

                ax.set_xticks(xtick)
                ax.set_yticks(np.arange(-3, 3, 0.5))
                ax.tick_params(axis="x", labelsize=7)
                ax.tick_params(axis="y", labelsize=7)

                ax.minorticks_on()
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))

                ax.set_ylim(-3, 3)
                ax.set_xlim( 0, time_samples[-1] + 1 / ecg_sampling_frequency, )

                if grid_on:
                    ax.grid( which="major", linestyle="-", linewidth=0.5, color="#A9A9A9", )

                    ax.grid( which="minor", linestyle="-", linewidth=0.5, color="#D3D3D3", )

        fig.text(0.5, 0.04, "(s)", ha="center")
        fig.text( 0.04, 0.5, "(mV)", va="center", rotation="vertical", )

        if save_figs_folder is not None:
            baseline_filename = ( "_baselineRemoved" if remove_baseline else "" )

            if save_figs_folder.endswith(".pdf"):
                additional_part = ( f"_{ecg_real_diagnosis}" f"_VS_{ecg_gen_diagnosis}" f"{baseline_filename}_IoU.pdf" )

                figtitle = re.sub( r"\.pdf$", additional_part, save_figs_folder, )

            else:
                plot_type = "BANDS" if bands else "SAMPLES"

                figtitle = os.path.join( save_figs_folder, f"real_{ecg_real_diagnosis}"
                    f"_VS_synthFrom_{ecg_gen_diagnosis}" f"_ecg_{plot_type}"
                    f"{baseline_filename}" f"_IoU_{len(ecg_gen_sample)}elem.pdf", )

            plt.rc("pdf", fonttype=42)
            plt.rc("ps", fonttype=42)

            fig.savefig( figtitle, transparent=False, bbox_inches=None, pad_inches=0, )

            plt.close(fig)

        elif show_plots:
            plt.show()

        else:
            plt.close(fig)

    return results