import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--task_config', type=str, default='motion_blur', help='Task configuration')
    args = parser.parse_args()

    path = f'./results/{args.task_config}/'

    print('path = ', path)

    # Load results

    lpips = np.load(path + 'pathwise_lpips.npy')
    psnr = np.load(path + 'pathwise_psnr.npy')
    distances = np.load(path + 'pathwise_distances.npy')

    n_data_samples, n_max = distances.shape

    n_path_groups = 10

    lpips_best = np.zeros([n_data_samples, n_max])
    psnr_best = np.zeros([n_data_samples, n_max])
    distances_best = np.zeros([n_data_samples, n_max])

    for n in range(n_max):

        for _ in range(n_path_groups):

            # Sample n paths without replacement from n_max
            paths = np.random.choice(n_max, n+1, replace=False)

            best_paths = np.argmin(distances[:, paths], axis=1)

            print('best_paths:', best_paths)

            print(f"Best path for {n+1} paths: {best_paths}")
            print(f"PSNR: {psnr[:, best_paths]}")
            print(f"LPIPS: {lpips[:, best_paths]}")
            print(f"Distance: {distances[:, best_paths]}")
            print(f"Paths: {paths}")

            for img_idx in range(n_data_samples):

                lpips_best[img_idx, n] += lpips[img_idx, best_paths[img_idx]]
                psnr_best[img_idx, n] += psnr[img_idx, best_paths[img_idx]]
                distances_best[img_idx, n] += distances[img_idx, best_paths[img_idx]]

        lpips_best[:, n] /= n_path_groups
        psnr_best[:, n] /= n_path_groups
        distances_best[:, n] /= n_path_groups


    net_lpips = np.mean(lpips_best, axis=0)
    net_psnr = np.mean(psnr_best, axis=0)
    net_distances = np.mean(distances_best, axis=0)

    np.save(path + 'best_of_n_lpips.npy', net_lpips)
    np.save(path + 'best_of_n_psnr.npy', net_psnr)
    np.save(path + 'best_of_n_distances.npy', net_distances) 


    path_count = np.arange(1, n_max+1)

    # Plot results on 3 axes

    plt.plot(path_count, net_psnr, marker='o', lw=2, color='k', label='PSNR')
    plt.xlabel('Number of paths')
    plt.ylabel('Metric')
    plt.legend()

    plt.savefig(path + 'best_of_n_psnr.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.plot(path_count, net_lpips, marker='o', lw=2, color='k', label='LPIPS')
    plt.xlabel('Number of paths')
    plt.ylabel('Metric')
    plt.legend()

    plt.savefig(path + 'best_of_n_lpips.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.plot(path_count, net_distances, marker='o', lw=2, color='k', label='Distance')
    plt.xlabel('Number of paths')
    plt.ylabel('Metric')
    plt.legend()

    plt.savefig(path + 'best_of_n_distances.png', dpi=500, bbox_inches='tight')
    plt.close()

      




    







        










