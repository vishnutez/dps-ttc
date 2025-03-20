import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def compute_semantic_distance(real_image, fake_image):
    # Load pre-trained model
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device='cuda')

    real_images = mtcnn(real_image).unsqueeze(0)
    fake_images = mtcnn(fake_image).unsqueeze(0)

    real_embeddings = resnet(real_images)
    fake_embeddings = resnet(fake_images)

    diff = real_embeddings - fake_embeddings

    # Compute distance
    distance = torch.linalg.norm(diff, dim=-1)

    return distance.detach().cpu().item()


def compute_lpips(real_images, fake_images, net_type='squeeze'):

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)

    real_images_float = to_normalized_float(real_images)
    fake_images_float = to_normalized_float(fake_images)

    lpips.to(real_images.device)

    lpips.update(real_images_float, fake_images_float)

    lpips_val = lpips.compute()
    lpips.reset()

    return lpips_val.item()


def compute_psnr(real_images, fake_images):

    from torchmetrics.image import PeakSignalNoiseRatio

    psnr = PeakSignalNoiseRatio()

    psnr.to(real_images.device)

    psnr.update(fake_images, real_images)

    psnr_val = psnr.compute()
    psnr.reset()

    return psnr_val.item()


def compute_psnr_manual(real_images, fake_images):

    mse = torch.mean((real_images - fake_images) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr.item()


# Function to convert a Image into tensor of unit8 type

def to_int(image):

    return (image * 255).type(torch.uint8)


def to_normalized_float(image):  # convert to [-1, 1]

    image_min = image.min()
    image_max = image.max()

    image = (image - image_min) / (image_max - image_min)

    return (image * 2 - 1).type(torch.float32)


def load_images_from_path(path, n_samples):
    
    from glob import glob
    from PIL import Image

    # Read all images from folder
    image_paths = sorted(glob(path + "/*.png", recursive=True))  # Change extension if needed
    images = [Image.open(image_paths[i]).convert('RGB') for i in range(n_samples)]

    return images


def compute_relevant_metrics(real_image, fake_image, verbose=False):

    import torchvision.transforms as transforms

    lpips = compute_lpips(transforms.ToTensor()(real_image).unsqueeze(0), transforms.ToTensor()(fake_image).unsqueeze(0))
    psnr = compute_psnr(transforms.ToTensor()(real_image).unsqueeze(0), transforms.ToTensor()(fake_image).unsqueeze(0))
    semantic_distance = compute_semantic_distance(real_image, fake_image)  # Real and fake images are Pillow Images

    if verbose:
        print(f"PSNR (H): {psnr} \t LPIPS (L): {lpips} \t Semantic Distance (L): {semantic_distance}")

    return  psnr, lpips, semantic_distance


if __name__ == '__main__':

    semantic_scale = 0.01
    measurement_scale = 0.3

    # dir = f'results_ood/motion_blur_noise_sigma_0.05_1x_dps_scale_1.0'
    dir = f'results_semantic_norm2_anneal_10x/motion_blur_semantic_{semantic_scale}_measurement_{measurement_scale}'

    import os

    real_image = Image.open(os.path.join(dir, 'label', '00004.png')).convert('RGB')
    img_path = os.path.join(dir, 'recon_paths', '00004')

    metrics = []

    for i in range(len(os.listdir(img_path))):
        curr_img_path = os.path.join(img_path, f'path#{i}.png')
        fake_image = Image.open(curr_img_path).convert('RGB')
        psnr, lpips, semantic_distance = compute_relevant_metrics(real_image, fake_image, verbose=True)
        metrics.append([psnr, lpips, semantic_distance])
    
    # Save as csv
    import pandas as pd
    
    df = pd.DataFrame(metrics, columns=['PSNR', 'LPIPS', 'Semantic Distance'])
    df.to_csv(f'metrics/norm2_anneal5x_semantic_{semantic_scale}_measurement_{measurement_scale}_metrics.csv', index=False)


