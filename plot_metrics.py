import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results_hyper/metrics.csv', header=0, index_col=None)  # Ensure the first column is not treated as an index

df_dps = df[df['semantic_scale'] == 0]

# Compute averages
average_psnr_dps = df_dps['psnr'].mean()
average_lpips_dps = df_dps['lpips'].mean()
average_measurement_distance_dps = df_dps['measurement_distance'].mean()
average_semantic_distance_dps = df_dps['semantic_distance'].mean()

semantic_scale = 0.3
anneal_factor = 1

df_scale_0_3 = df[(df['semantic_scale'] == semantic_scale) & (df['anneal_factor'] == anneal_factor) & (df['n_guid_images'] == 1)]

# Compute averages
average_psnr_0_3 = df_scale_0_3['psnr'].mean()
average_lpips_0_3 = df_scale_0_3['lpips'].mean()
average_measurement_distance_0_3 = df_scale_0_3['measurement_distance'].mean()
average_semantic_distance_0_3 = df_scale_0_3['semantic_distance'].mean()

# Prepare data for the bar plot
categories = ['PSNR', 'LPIPS', 'Measurement Distance', 'Semantic Distance']
values_dps = [average_psnr_dps, average_lpips_dps, average_measurement_distance_dps, average_semantic_distance_dps]
values_0_3 = [average_psnr_0_3, average_lpips_0_3, average_measurement_distance_0_3, average_semantic_distance_0_3]


# Create a figure with subplots for each metric
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

metrics = ['PSNR', 'LPIPS', 'Measurement Distance', 'Semantic Distance']
values_dps = [average_psnr_dps, average_lpips_dps, average_measurement_distance_dps, average_semantic_distance_dps]
values_0_3 = [average_psnr_0_3, average_lpips_0_3, average_measurement_distance_0_3, average_semantic_distance_0_3]

# Print metrics side by side
print(f"{'Metric':<25}{'Semantic Scale = 0':<20}{'Semantic Scale = {semantic_scale}':<20}")
print(f"{'-'*65}")
for metric, value_dps, value_0_3 in zip(metrics, values_dps, values_0_3):
    print(f"{metric:<25}{value_dps:<20.4f}{value_0_3:<20.4f}")

for i, ax in enumerate(axes):
    ax.bar(['Semantic Scale = 0', 'Semantic Scale = {semantic_scale}'], [values_dps[i], values_0_3[i]], color=['tab:blue', 'tab:orange'])
    ax.set_title(metrics[i])
    ax.set_ylabel('Average Value')
    ax.set_xlabel('Semantic Scale')

# Adjust layout and save the figure
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/comparison_metrics_separate_axes_tab.png', bbox_inches='tight')
plt.close()








