import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap


# Load the data from each of the tasks and plot the results

if __name__ == '__main__':


    # Load the data

    tasks = ['inpainting', 'super_resolution', 'motion_blur', 'phase_retrieval']

    tasks_to_names = {
        'inpainting': 'Inpainting',
        'super_resolution': 'Super Resolution',
        'phase_retrieval': 'Phase Retrieval',
        'motion_blur': 'Motion Blur'
    }

    tasks_to_colors = {
        'inpainting': 'tab:purple',
        'super_resolution': 'tab:red',
        'phase_retrieval': 'tab:blue', 
        'motion_blur': 'tab:orange'
    }

    tasks_to_markers = {
        'inpainting': 'o',
        'super_resolution': 's',
        'phase_retrieval': '^',
        'motion_blur': 'x'
    }

    # Plot the results

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i, task in enumerate(tasks):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        psnr = np.load(f'./results/{task}/best_of_n_psnr.npy')
        lpips = np.load(f'./results/{task}/best_of_n_lpips.npy')
        distances = np.load(f'./results/{task}/best_of_n_distances.npy')
        n = np.arange(1, len(psnr)+1)

        # Using sns
        ax[0].plot(n, psnr, label=f'{tasks_to_names[task]}', color=tasks_to_colors[task], marker=tasks_to_markers[task])
        ax[0].set_title('PSNR')
        ax[0].set_xlabel('Number of samples')
        ax[0].set_ylabel('Metric')
        ax[0].set_xticks(n)

        ax[1].plot(n, lpips, color=tasks_to_colors[task], marker=tasks_to_markers[task])
        ax[1].set_title('LPIPS')
        ax[1].set_xlabel('Number of samples')
        ax[1].set_xticks(n)

        ax[2].plot(n, distances, color=tasks_to_colors[task], marker=tasks_to_markers[task])
        ax[2].set_title('Distance')
        ax[2].set_xlabel('Number of samples')
        ax[2].set_xticks(n)
    
        # Create a common legend for all the plots

        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.tight_layout()
        plt.savefig(f'./results/plot_{task}.png', bbox_inches='tight')
        plt.close()

    



