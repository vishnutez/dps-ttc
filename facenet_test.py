from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
import torch
from PIL import Image
import os

import matplotlib.pyplot as plt

from util.img_utils import clear_color


img_size = 256
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=img_size, margin=30, min_face_size=20)  # Keep all parameters default

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()


# # Load images
# img1 = Image.open(os.path.join('../facedata-preprocessed/0004.png'))
# img1_crop = mtcnn(img1, save_path='../cropped-data/fn-0004.png')
# img1_emb = resnet(img1_crop.unsqueeze(0))

# img2 = Image.open(os.path.join('../facedata-preprocessed/0003.png'))
# img2_crop = mtcnn(img2, save_path='../cropped-data/fn-0003.png')
# img2_emb = resnet(img2_crop.unsqueeze(0))

# img3 = Image.open(os.path.join('../data/ffhq256/00000.png'))
# img3_crop = mtcnn(img3, save_path='../cropped-data/ffhq-00000.png')
# img3_emb = resnet(img3_crop.unsqueeze(0))


# # Compute the Euclidean distance between the two embeddings
# distance = (img1_emb - img2_emb).norm().item()
# print(f"Distance of a same person = {distance}")

# other_distance = (img1_emb - img3_emb).norm().item()
# print(f"Distance of a different person = {other_distance}")


# Ref image
ref_img_pil = Image.open(os.path.join('../facedata-preprocessed/0003.png'))

import torchvision.transforms as transforms

# Convert to tensor

# Convert to Floattensor

ref_img_np = np.uint8(ref_img_pil)

ref_img_tensor = torch.from_numpy(ref_img_np).unsqueeze(0).type(torch.float32)

print(f'ref_img_tensor = {ref_img_tensor}')

# plt.imsave('../cropped-data/fn-0003-tensor.png', clear_color(ref_img.unsqueeze(0)))

ref_img_cropped = mtcnn(ref_img_tensor)

print(f'ref_img_cropped = {ref_img_cropped.shape}')

# plt.imsave('../cropped-data/fn-0003-cropped-tensor.png', clear_color(ref_img_cropped.unsqueeze(0)))

ref_img_emb = resnet(ref_img_cropped)

# sigma = 0.1

# # Test on multiple images
# same_faces =  sorted(os.listdir('../facedata-preprocessed'))
# for i in range(4):
#     noise = torch.randn_like(ref_img_cropped)
#     file = same_faces[i]
#     img = Image.open(os.path.join('../facedata-preprocessed', file))
    
#     img_crop = mtcnn(img, save_path=f'../cropped-data/fn-{file}')
#     print(f'img = {img_crop.shape}')
#     noisy_img = img_crop + sigma * noise
#     img_emb = resnet(img_crop.unsqueeze(0))
#     distance = (ref_img_emb - img_emb).norm().item()
#     print(f"Distance of a same person = {distance}")

#     noisy_img_emb = resnet(noisy_img.unsqueeze(0))
#     distance = (ref_img_emb - noisy_img_emb).norm().item()
#     print(f"Distance of a same person with noise = {distance}")


# steps = range(990, 0, -10)
# distances = []
# for i in steps:
#     file_name = str(i).zfill(4)
#     img = Image.open(os.path.join(f'./results_ood/super_resolution_noise_sigma_0.05_dps_scale_0.3/progress/x_{file_name}.png'))

#     # Convert to tensor
#     img = img.convert('RGB')
#     # print(f'img = {img.size}')
#     img_crop = mtcnn(img, save_path=f'../cropped-data-progress/fn-{file_name}.png')
    
# #     noisy_img = img_crop + sigma * noise
#     img_emb = resnet(img_crop.unsqueeze(0))
#     distance = (ref_img_emb - img_emb).norm().item()
#     distances.append(distance)

# import matplotlib.pyplot as plt
# plt.plot(steps, distances)
# plt.show()
# plt.savefig('distances.png')




# n_images = 10
# # Different images

# diff_faces =  sorted(os.listdir('../data/ffhq256'))
# for i in range(n_images):
#     file = diff_faces[i]
#     img = Image.open(os.path.join('../data/ffhq256', file))
#     img_crop = mtcnn(img, save_path=f'../cropped-data/ffhq-{file}')
#     img_emb = resnet(img_crop.unsqueeze(0))
#     distance = (ref_img_emb - img_emb).norm().item()
#     print(f"Distance of a different person = {distance}")


# rand_img = torch.randn_like(ref_img_cropped)
# rand_img_emb = resnet(rand_img.unsqueeze(0))
# distance = (ref_img_emb - rand_img_emb).norm().item()
# print(f"Distance of a random image = {distance}")
    




