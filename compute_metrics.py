import torch

def compute_fid(real_images, fake_images, num_features=2048):

    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=num_features, reset_real_features=False)

    real_images_int = to_int(real_images)
    fake_images_int = to_int(fake_images)

    fid.update(real_images_int, real=True)
    fid.update(fake_images_int, real=False)

    fid_val = fid.compute()
    fid.reset()

    return fid_val


def compute_lpips(real_images, fake_images, net_type='squeeze'):

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)

    real_images_float = to_normalized_float(real_images)
    fake_images_float = to_normalized_float(fake_images)

    lpips.to(real_images.device)

    lpips.update(real_images_float, fake_images_float)

    lpips_val = lpips.compute()
    lpips.reset()

    return lpips_val


def compute_psnr(real_images, fake_images):

    from torchmetrics.image import PeakSignalNoiseRatio

    psnr = PeakSignalNoiseRatio()

    psnr.to(real_images.device)

    psnr.update(fake_images, real_images)

    psnr_val = psnr.compute()
    psnr.reset()

    return psnr_val


def compute_psnr_manual(real_images, fake_images):

    mse = torch.mean((real_images - fake_images) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr


# Function to convert a Image into tensor of unit8 type

def to_int(image):

    return (image * 255).type(torch.uint8)


def to_normalized_float(image):

    image_min = image.min()
    image_max = image.max()

    image = (image - image_min) / (image_max - image_min)

    return (image * 2 - 1).type(torch.float32)



def load_images_from_path(path, N):

    from glob import glob
    from PIL import Image
    import torchvision.transforms as transforms

    # Read all images from folder
    image_paths = sorted(glob(path + "/*.png", recursive=True))  # Change extension if needed
    images = [transforms.ToTensor()(Image.open(image_paths[i]).convert('RGB')) for i in range(N)]

    return torch.stack(images)


def compute_all_metrics(real_images, fake_images, verbose=False):

    fid = compute_fid(real_images, fake_images)
    lpips = compute_lpips(real_images, fake_images)
    psnr = compute_psnr(real_images, fake_images)

    if verbose:
        print(f"FID (L): {fid} \t LPIPS (L): {lpips} \t PSNR (H): {psnr}")

    return fid, lpips, psnr

if __name__ == '__main__':

    n_samples = 3
    img_size = 256

    # Load images

    real_images = load_images_from_path(path="../data/ffhq256/", N=n_samples)
    print('Real images = ', real_images.shape)


    # Load fake images from blind-dps
    print('Compare with reconstructions')
    fake_images = load_images_from_path(path="./results/blind_blur/recon/", N=n_samples)
    fid, lpips, psnr = compute_all_metrics(real_images, fake_images, verbose=True)

    # Load fake images from blind-dps inputs
    print('Compare with inputs')
    fake_images = load_images_from_path(path="./results/blind_blur/input/", N=n_samples)
    fid, lpips, psnr = compute_all_metrics(real_images, fake_images, verbose=True)


    # # Compute PSNR manually
    # psnr_manual = compute_psnr_manual(real_images, fake_images)
    # print(f"Computed PSNR manually: {psnr_manual}")
