import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results_fs/metrics.csv', header=0, index_col=None)  # Ensure the first column is not treated as an index

print('df = \n', df)

df_dps = df[df['semantic_scale'] == 0]

# Compute averages
average_psnr_dps = df_dps['psnr'].mean()
average_lpips_dps = df_dps['lpips'].mean()
average_measurement_distance_dps = df_dps['measurement_distance'].mean()
average_semantic_distance_dps = df_dps['semantic_distance'].mean()

# Compute standard deviations
std_psnr_dps = df_dps['psnr'].std()
std_lpips_dps = df_dps['lpips'].std()
std_measurement_distance_dps = df_dps['measurement_distance'].std()
std_semantic_distance_dps = df_dps['semantic_distance'].std()

semantic_scale = 0.2
anneal_factor = 1

df_dps_fs = df[(df['semantic_scale'] == semantic_scale) & (df['anneal_factor'] == anneal_factor)]

# Compute averages
average_psnr_dps_fs = df_dps_fs['psnr'].mean()
average_lpips_dps_fs = df_dps_fs['lpips'].mean()
average_measurement_distance_dps_fs = df_dps_fs['measurement_distance'].mean()
average_semantic_distance_dps_fs = df_dps_fs['semantic_distance'].mean()

# Compute standard deviations
std_psnr_dps_fs = df_dps_fs['psnr'].std()
std_lpips_dps_fs = df_dps_fs['lpips'].std()
std_measurement_distance_dps_fs = df_dps_fs['measurement_distance'].std()
std_semantic_distance_dps_fs = df_dps_fs['semantic_distance'].std()

# Prepare data for the bar plot
categories = ['PSNR', 'LPIPS', 'Measurement Distance', 'Semantic Distance']
values_dps = [average_psnr_dps, average_lpips_dps, average_measurement_distance_dps, average_semantic_distance_dps]
values_dps_fs = [average_psnr_dps_fs, average_lpips_dps_fs, average_measurement_distance_dps_fs, average_semantic_distance_dps_fs]


# Create a figure with subplots for each metric
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

metrics = ['PSNR', 'LPIPS', 'Measurement Distance', 'Semantic Distance']
values_dps = [average_psnr_dps, average_lpips_dps, average_measurement_distance_dps, average_semantic_distance_dps]
values_dps_fs = [average_psnr_dps_fs, average_lpips_dps_fs, average_measurement_distance_dps_fs, average_semantic_distance_dps_fs]

# Print metrics and standard deviations side by side
print(f"{'Metric':<25}{'DPS (Mean ± Std)':<30}{f'DPS+FS ({semantic_scale}x{anneal_factor}) (Mean ± Std)':<30}")
print(f"{'-'*85}")
for metric, mean_dps, std_dps, mean_dps_fs, std_dps_fs in zip(
    metrics, values_dps, 
    [std_psnr_dps, std_lpips_dps, std_measurement_distance_dps, std_semantic_distance_dps],
    values_dps_fs, 
    [std_psnr_dps_fs, std_lpips_dps_fs, std_measurement_distance_dps_fs, std_semantic_distance_dps_fs]
):
    print(f"{metric:<25}{mean_dps:.4f} ± {std_dps:.4f} \t {mean_dps_fs:.4f} ± {std_dps_fs:.4f}")

for i, ax in enumerate(axes):
    ax.plot([0, 1], [values_dps[i], values_dps_fs[i]], color='tab:blue', marker='o', label=f'DPS vs DPS+FS={semantic_scale}x{anneal_factor}')
    # ax.plot([1, ], color='tab:orange', marker='o', label=f'')
    ax.set_title(metrics[i])
    ax.set_ylabel('Metrics')
    ax.set_xlabel('Methods')
    ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/comparison_metrics_separate_axes_tab_line.png', bbox_inches='tight')
plt.close()

# print('Here: ', df['guid_images'])







