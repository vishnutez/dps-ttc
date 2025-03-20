from compute_metrics import compute_semantic_distance
from PIL import Image

# Read the images

ref_img = Image.open('./results_ood/motion_blur_noise_sigma_0.05_1x_dps_semantic_scale_0.3_guid_scale_1.0/label/00004.png')
ref_img = ref_img.convert('RGB')


img = Image.open('./results_ood/motion_blur_noise_sigma_0.05_1x_dps_semantic_scale_0.3_guid_scale_1.0/recon_paths/00004/path#1.png')
img = img.convert('RGB')


# Compute the semantic distance
semantic_distance = compute_semantic_distance(ref_img, img)
print('Semantic Distance = ', semantic_distance)

