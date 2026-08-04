import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import random
import os


def plot_12lead_ecg(ecg, fs, lead_names=None, title=None, save_path=None, show_plot = False ):
    
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    
    ecg = np.asarray(ecg)

    if ecg.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {ecg.shape}")

    if ecg.shape[0] == 12:
        ecg_12 = ecg
    elif ecg.shape[1] == 12:
        ecg_12 = ecg.T
    else:
        raise ValueError(f"Expected shape (12, n_samples) or (n_samples, 12), got {ecg.shape}")

    n_leads, n_samples = ecg_12.shape
    if n_leads != 12:
        raise ValueError(f"Expected 12 leads, got {n_leads}")

    t = np.arange(n_samples) / fs
    t0 = 0
    t1 = n_samples / fs

    max_abs = np.nanmax(np.abs(ecg_12))
    if not np.isfinite(max_abs) or max_abs == 0:
        y_lim = 0.5
    else:
        y_lim = np.ceil(max_abs / 0.5) * 0.5

    #Rescale fig width according to ECG length
    ecg_seconds = n_samples / fs
    if np.isclose(ecg_seconds, 3.0):
        fig_width = 16
    else:
        fig_width = 16 * (ecg_seconds / 3.0)
    
    figsize=(fig_width, 8)

    # fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True)
    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0, "wspace": 0.01})
    
    fig.patch.set_facecolor("white")

    for i in range(12):
        row = i % 3
        col = i // 3
        ax = axes[row, col]
        
        # ax.set_aspect(0.4, adjustable="box")
        # ax.set_box_aspect(0.4)

        ax.plot(t, ecg_12[i], color="black", linewidth=0.9)
        ax.set_title(lead_names[i], fontsize=11, loc="left")
        
        ax.set_xlim(t0, t1)
        ax.set_ylim(-y_lim, y_lim)
        
        # ECG paper grid:
        # small box: 0.04 s x 0.1 mV
        # big box:   0.20 s x 0.5 mV
        ax.xaxis.set_major_locator(MultipleLocator(0.20))  # large ECG box
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))  # small ECG box

        ax.yaxis.set_major_locator(MultipleLocator(0.50))  # large ECG box
        ax.yaxis.set_minor_locator(MultipleLocator(0.10))  # small ECG box

        ax.grid(which="major", color="#e8a0a0", linewidth=0.8)
        ax.grid(which="minor", color="#f9dddd", linewidth=0.3)
        
        # Label only whole seconds on x-axis, but keep 0.20 s major grid lines
        def x_formatter(x, pos):
            if x < -1e-9:
                return ""
            if abs((x / 0.4) - round(x / 0.4)) < 1e-9:
                return f"{x:.1f}".rstrip("0").rstrip(".")
            return ""
        
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))

        # Label y every 0.5 mV
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))



        if row < 2:
            ax.tick_params(labelbottom=False)
        if col > 0:
            ax.tick_params(labelleft=False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
        ax.tick_params(axis="x", labelbottom=True, labelsize=5)

    for ax in axes[:, 0]:
        ax.set_ylabel("mV")

    if title is not None:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle("12-lead ECG", fontsize=12)
        
    
    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    
    # plt.tight_layout()
    if show_plot:
        plt.show()
    
    plt.close(fig)
    return




def plot_two_12lead_ecg(ecg, ecg_2, fs, lead_names=None, title=None, save_path=None, show_plot=False, ecg_labels = None):
    
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    
    ecg = np.asarray(ecg)
    ecg_2 = np.asarray(ecg_2)
    
    if ecg_labels is not None:
        if len(ecg_labels) != 2:
            raise ValueError(f"Expected ecg_labels to have length 2, got {len(ecg_labels)}")
        label_1, label_2 = ecg_labels
    else:
        label_1, label_2 = "ECG 1", "ECG 2"

    if ecg.ndim != 2:
        raise ValueError(f"Expected a 2D array for ecg, got shape {ecg.shape}")

    if ecg.shape[0] == 12:
        ecg_12 = ecg
    elif ecg.shape[1] == 12:
        ecg_12 = ecg.T
    else:
        raise ValueError(f"Expected ecg shape (12, n_samples) or (n_samples, 12), got {ecg.shape}")

    if ecg_2.ndim != 2:
        raise ValueError(f"Expected a 2D array for ecg_2, got shape {ecg_2.shape}")

    if ecg_2.shape[0] == 12:
        ecg_12_2 = ecg_2
    elif ecg_2.shape[1] == 12:
        ecg_12_2 = ecg_2.T
    else:
        raise ValueError(f"Expected ecg_2 shape (12, n_samples) or (n_samples, 12), got {ecg_2.shape}")

    n_leads, n_samples = ecg_12.shape
    if n_leads != 12:
        raise ValueError(f"Expected 12 leads, got {n_leads}")

    n_leads_2, n_samples_2 = ecg_12_2.shape
    if n_leads_2 != 12:
        raise ValueError(f"Expected 12 leads in ecg_2, got {n_leads_2}")

    if n_samples_2 != n_samples:
        raise ValueError(f"Expected both ECGs to have the same number of samples, got {n_samples} and {n_samples_2}")

    t = np.arange(n_samples) / fs
    t0 = 0
    t1 = n_samples / fs

    max_abs = np.nanmax(np.abs([ecg_12, ecg_12_2]))
    if not np.isfinite(max_abs) or max_abs == 0:
        y_lim = 0.5
    else:
        y_lim = np.ceil(max_abs / 0.5) * 0.5

    #Rescale fig width according to ECG length
    ecg_seconds = n_samples / fs
    if np.isclose(ecg_seconds, 3.0):
        fig_width = 16
    else:
        fig_width = 16 * (ecg_seconds / 3.0)
    
    figsize = (fig_width, 8)

    # fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True)
    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0, "wspace": 0.01})
    
    fig.patch.set_facecolor("white")

    for i in range(12):
        row = i % 3
        col = i // 3
        ax = axes[row, col]
        
        # ax.set_aspect(0.4, adjustable="box")
        # ax.set_box_aspect(0.4)

        ax.plot(t, ecg_12[i], color="black", linewidth=0.9, label=label_1)
        ax.plot(t, ecg_12_2[i], color="red", linewidth=0.9, label=label_2)
        ax.set_title(lead_names[i], fontsize=11, loc="left")
        
        ax.set_xlim(t0, t1)
        ax.set_ylim(-y_lim, y_lim)
        
        # ECG paper grid:
        # small box: 0.04 s x 0.1 mV
        # big box:   0.20 s x 0.5 mV
        ax.xaxis.set_major_locator(MultipleLocator(0.20))  # large ECG box
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))  # small ECG box

        ax.yaxis.set_major_locator(MultipleLocator(0.50))  # large ECG box
        ax.yaxis.set_minor_locator(MultipleLocator(0.10))  # small ECG box

        ax.grid(which="major", color="#e8a0a0", linewidth=0.8)
        ax.grid(which="minor", color="#f9dddd", linewidth=0.3)
        
        # Label only whole seconds on x-axis, but keep 0.20 s major grid lines
        def x_formatter(x, pos):
            if x < -1e-9:
                return ""
            if abs((x / 0.4) - round(x / 0.4)) < 1e-9:
                return f"{x:.1f}".rstrip("0").rstrip(".")
            return ""
        
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))

        # Label y every 0.5 mV
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))



        if row < 2:
            ax.tick_params(labelbottom=False)
        if col > 0:
            ax.tick_params(labelleft=False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
        ax.tick_params(axis="x", labelbottom=True, labelsize=5)

    for ax in axes[:, 0]:
        ax.set_ylabel("mV")

    if title is not None:
        fig.suptitle(title, fontsize=12, y=0.995)
    else:
        fig.suptitle("12-lead ECG", fontsize=12, y=0.995)
    
    legend_handles = [ Line2D([0], [0], color="black", linewidth=0.9, label=ecg_labels[0]), Line2D([0], [0], color="red", linewidth=0.9, label=ecg_labels[1]), ]

    fig.legend( handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=8, frameon=False )
    fig.subplots_adjust(top=0.92)
    
    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    
    # plt.tight_layout()
    if show_plot:
        plt.show()
    
    plt.close(fig)
    return


def get_paired_plots(ecg_datasets, real_fake_labels = None,  rand_index = False, given_index = 0, seed = 42, 
                     show_plot = True, save_folder = None, sampling_frq = 500):
    '''Get paired ECG plots for real and fake data.
    real_fake_labels: [(real_lab, fake_lab), ...] list of tuples of real and fake labels to plot
    
    '''
    random.seed(seed)
    
    if real_fake_labels is None:
        real_fake_labels = [(dataset.real.label, dataset.fake.label) for dataset in ecg_datasets]
    
    for real_label, fake_label in real_fake_labels:
        
        for pair in ecg_datasets:
            if ( pair.real.label == real_label and pair.fake.label == fake_label ):
                print(f"Plotting {real_label} to {fake_label}...")
                if rand_index:
                    index = np.min([random.randint(0, len(pair.real.data)-1), random.randint(0, len(pair.fake.data)-1)])
                else:
                    index = given_index

                real_data = pair.real.data[index]
                fake_data = pair.fake.data[index]

                plot_title = f"{pair.real.label.upper()}_to_{pair.fake.label.upper()}_ind{index:04d}"


                if save_folder is not None:
                    save_name = plot_title + ".pdf"
                    save_full_path = os.path.join(save_folder, save_name)

    
                plot_two_12lead_ecg(real_data, fake_data, 
                                    title = plot_title, 
                                    ecg_labels = [pair.real.label, pair.fake.label],
                                    fs = sampling_frq, show_plot=show_plot, 
                                    save_path = save_full_path if save_folder is not None else None)
                print("--------------------------------------------------------------------------------------------")