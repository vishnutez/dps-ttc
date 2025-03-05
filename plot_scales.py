import numpy as np
import matplotlib.pyplot as plt


sampler_names = {
    'dps': 'DPS',
    'dps_anneal': 'DPS Anneal',
}

# Plot DPS scales
sampler = 'dps'
guidance_scales = np.load(f'./results_annealing_exp/motion_blur_noise_sigma_0.05_{sampler}/{sampler}_guidance_scales.npy')
num_steps = guidance_scales.shape[1]
steps = np.arange(num_steps-1, -1, -1)
for n in range(len(guidance_scales)):
    plt.plot(steps, guidance_scales[n], label=f'{sampler_names[sampler]}-Path: {n+1}')


# Plot DPS Anneal scales
sampler = 'dps_anneal'
guidance_scales = np.load(f'./results_annealing_exp/motion_blur_noise_sigma_0.05_{sampler}/{sampler}_guidance_scales.npy')
num_steps = guidance_scales.shape[1]
for n in range(len(guidance_scales)):
    plt.plot(steps, guidance_scales[n], label=f'{sampler_names[sampler]}-Path: {n+1}')


plt.xlabel('Steps')
plt.ylabel('Guidance Scale')
plt.legend()
plt.show()
plt.savefig(f'comparison_of_guidance_scales_for_mb.png')
