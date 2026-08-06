import numpy as np
import umap
import os


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats.kde import gaussian_kde
import tqdm

from make_data import ECGDataPair, ECGSample


def plot_UMAP_mesh(real_data, gen_data, real_class, gen_from_class, ax2, fig, title_on = True, save_single = None):
    '''
    Plot UMAP 2d mesh of real data and scatter of synthetic data.
    '''
    
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    x = real_data[:, 0]
    y = real_data[:, 1]

    mesh_min_x = np.min([x.min(), np.min(np.squeeze(gen_data[:, 0]))])
    mesh_max_x = np.max([x.max(), np.max(np.squeeze(gen_data[:, 0]))])
    mesh_min_y = np.min([y.min(), np.min(np.squeeze(gen_data[:, 1]))])
    mesh_max_y = np.max([y.max(), np.max(np.squeeze(gen_data[:, 1]))])

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[mesh_min_x:mesh_max_x:x.size**0.5*1j,mesh_min_y:mesh_max_y:y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Scatter plot

    im = ax2.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1)
    ax2.scatter(gen_data[:,0], gen_data[:,1], c='m', alpha=0.5, linewidth=0.1, marker='*', label='Synthetic')
    
    if title_on:
        ax2.set_title(f'Real {real_class} heatmap, synth (from {gen_from_class} ECG) in scatter ',
                    fontsize=15,
                    pad=10)
        # fig.colorbar(im, ax=ax2)
        # ax2.legend()
        
    if save_single is not None:
        save_name = os.path.join(save_single, f'umap_{real_class}_from_{gen_from_class}.pdf')
        fig.savefig( save_name, dpi=300, bbox_inches='tight' )

    return ax2, fig



def plot_UMAP_datasets(umap_ecg_datapairs, title_on = True, save_path = None, save_single = None ):
    
    '''
    Takes as input the datapairs created in utils make_data_class.
    '''
    
    if len(umap_ecg_datapairs) == 0:
        raise ValueError("The list of ECG data pairs is empty.")
    
    n_plots = len(umap_ecg_datapairs)
    
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
        
        
    fig, axes = plt.subplots( nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False )   
    axes_flat = axes.flatten()

    for i, datapair in enumerate(umap_ecg_datapairs):
        real_data = datapair.real.data
        gen_data = datapair.fake.data
        real_class = datapair.real.label
        gen_from_class = datapair.fake.label
        
        plot_UMAP_mesh( real_data, gen_data, real_class, gen_from_class, axes_flat[i], fig, title_on=title_on, save_single=save_single )

    
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)
        
    if title_on:
        fig.suptitle('UMAP 2D projection of Real (mesh) and synthetic ECGs (scatter)', fontsize=20, y=1.02)
    
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  
    
    if save_path is not None:
        save_name = os.path.join(save_path, 'umap_ALL_datasets.pdf')
        fig.savefig( save_name, dpi=300, bbox_inches='tight' )
    
    return 



def make_2d_UMAP_datapairs(ecg_datasets, random_state = 42):
    '''
    Taskes as input the dataclass with the generated ECGs and make another dataclass where data is the 2d UMAP representation.
    This will serve as input for the plot_UMAP_datasets function.
    '''
    
    umap_ecg_datapairs = []
    
    for datapair in tqdm.tqdm(ecg_datasets):
        real_data = datapair.real.data
        gen_data = datapair.fake.data
        real_class = datapair.real.label
        gen_from_class = datapair.fake.label
        
        real_data_flat = real_data.reshape(real_data.shape[0], -1)
        gen_data_flat = gen_data.reshape(gen_data.shape[0], -1)
        
        # Fit UMAP on real data and transform both real and generated data
        reducer = umap.UMAP(random_state=random_state)
        real_2d = reducer.fit_transform(real_data_flat)
        gen_2d = reducer.transform(gen_data_flat)
        
        # Create new ECGDataPair with 2D UMAP data
        umap_datapair = ECGDataPair(
            real=ECGSample(label=real_class, data=real_2d),
            fake=ECGSample(label=gen_from_class, data=gen_2d)
        )
        
        umap_ecg_datapairs.append(umap_datapair)
    
    
    return umap_ecg_datapairs
    


