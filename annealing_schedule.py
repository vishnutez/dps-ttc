import numpy as np


num_timesteps = 1000

beta_min = 0.0001
beta_max = 0.02

beta_noising = np.linspace(beta_min, beta_max, num_timesteps)
beta_denoising = beta_noising[::-1]
meas_noise_std = 0.05

tau = 10

times = np.arange(num_timesteps-1, -1, -1)
print('times  = ', times)


anneal_loc = 0.5
anneal_scale = 10


t = 1 - times / num_timesteps
annealing = num_timesteps / (1 + np.exp(anneal_scale * (t - anneal_loc)))

net_scaling_multiplicative = beta_denoising / (annealing * meas_noise_std**2)

rescaled_annealing = 1 / (1 + np.exp(anneal_scale * (t - anneal_loc)))
net_scaling_additive = beta_denoising / (rescaled_annealing +  meas_noise_std**2)


import matplotlib.pyplot as plt

plt.plot(net_scaling_multiplicative)
plt.show()
plt.savefig('net_scaling_multiplicative.png')
plt.close()

plt.plot(net_scaling_additive)
plt.show()
plt.savefig('net_scaling_additive.png')
plt.close()

plt.plot(annealing)
plt.show()
plt.savefig('annealing.png')
plt.close()

plt.plot(rescaled_annealing)
plt.show()
plt.savefig('rescaled_annealing.png')
plt.close()

