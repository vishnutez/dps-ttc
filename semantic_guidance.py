import torch
import argparse
import os
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from util.img_utils import clear_color
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Semantic Guidance')
parser.add_argument('--guid_image_idx', type=int, default=3, help='Guidance image index')
parser.add_argument('--image_idx', type=int, default=4, help='Image index')
parser.add_argument('--n_steps', type=int, default=1000, help='Number of steps')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--n_paths', type=int, default=1, help='Number of paths')
parser.add_argument('--init_sigma', type=float, default=0.1, help='Initial exploration sigma')
parser.add_argument('--path_idx', type=int, default=7, help='Path index')

args = parser.parse_args()

os.makedirs('semantic_guidance', exist_ok=True)

for n in range(args.n_paths):
    os.makedirs(f'semantic_guidance/path_{n}', exist_ok=True)


guid_filename = str(args.guid_image_idx).zfill(4)
print('guid_filename:', guid_filename)
guid_image = Image.open(f'../facedata-preprocessed/{guid_filename}.png')

image_size = 256

mtcnn = MTCNN(image_size=image_size, margin=20, min_face_size=20)  # Keep everything default
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Get cropped and prewhitened image tensor

with torch.no_grad():
    guid_image_cropped = mtcnn(guid_image).unsqueeze(0)
    print('guid_image_cropped [-1, 1]:', guid_image_cropped.shape)
    guid_image_embedding = resnet(guid_image_cropped)
    print('guid_image_embedding shape:', guid_image_embedding.shape)
    # print('guid_image_embedding:', guid_image_embedding)

plt.imsave(f'semantic_guidance/guid_image_{guid_filename}.png', clear_color(guid_image_cropped))

# img_pil = Image.open('./results_ood/super_resolution_noise_sigma_0.05_8x_norm2_dps_semantic_scale_0.1_guid_scale_1.0/recon_paths/00004/path#1.png')

img_filename = str(args.image_idx).zfill(5)
img_pil = Image.open(f'./results_ood/super_resolution_noise_sigma_0.05_4x_dps_semantic_scale_0.8_guid_scale_0.01/recon_paths/{img_filename}/path#{args.path_idx}.png')

img = img_pil.convert('RGB')

x = mtcnn(img).unsqueeze(0)  # Obtained from MTCNN

# img_tensor = torch.as_tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
# x = (img_tensor / 255 - 0.5) / 0.5
print('img_tensor shape:', x.shape)

plt.imsave(f'semantic_guidance/image_{img_filename}_path#{args.path_idx}.png', clear_color(x))

# n_steps = 1000
# lr = 0.1


# print('x = ', x.shape)

print('Improving the image a bit by bit')

for t in range(args.n_steps):
    
    x.requires_grad_()
    z = resnet(x)
    print('z = ', z.shape)
    diff = z - guid_image_embedding
    distance = torch.linalg.norm(diff, ord=2, dim=1)
    grad_x = torch.autograd.grad(distance.sum(), x)[0]
    x = x - args.lr * grad_x
    x.detach_()
    if t % 200 == 0 or t==args.n_steps-1:
        for i in range(x.shape[0]):
            recon_image = clear_color(x[i].unsqueeze(0))
            plt.imsave(f'semantic_guidance/path_{i}/recon_image_{img_filename}_path#{args.path_idx}_step_{t}_path_{i}.png', recon_image)
    print('distance:', distance)


# recon_image = clear_color(x)
# plt.imsave(f'semantic_guidance/recon_image_{img_filename}_steps_{n_steps}_lr_{lr}.png', recon_image)

