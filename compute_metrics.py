import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def compute_semantic_distance(real_image, fake_image):
    # Load pre-trained model
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device='cuda')

    # # Compute embeddings
    # real_images = real_images.permute(0, 2, 3, 1).cpu().numpy()
    # fake_images = fake_images.permute(0, 2, 3, 1).cpu().numpy()

    # real_images = [Image.fromarray((real_images[i] * 255).astype('uint8')) for i in range(real_images.shape[0])]
    # fake_images = [Image.fromarray((fake_images[i] * 255).astype('uint8')) for i in range(fake_images.shape[0])]

    real_images = mtcnn(real_image).unsqueeze(0)
    fake_images = mtcnn(fake_image).unsqueeze(0)

    # print('real_images shape:', real_images.shape)
    # print('fake_images shape:', fake_images.shape)

    # real_images = torch.stack(real_images).to('cuda')
    # fake_images = torch.stack(fake_images).to('cuda')

    real_embeddings = resnet(real_images)
    fake_embeddings = resnet(fake_images)

    # print('real_embeddings shape:', real_embeddings.shape)
    # print('fake_embeddings shape:', fake_embeddings.shape)

    diff = real_embeddings - fake_embeddings

    # Compute distance
    distance = torch.linalg.norm(diff, dim=-1)

    return distance


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

def load_images_from_path_as_pil(path, N):
    
    from glob import glob
    from PIL import Image

    # Read all images from folder
    image_paths = sorted(glob(path + "/*.png", recursive=True))  # Change extension if needed
    images = [Image.open(image_paths[i]).convert('RGB') for i in range(N)]

    return images


def compute_relavent_metrics(real_image, fake_image, verbose=False):

    lpips = compute_lpips(real_image, fake_image)
    psnr = compute_psnr(real_image, fake_image)
    semantic_distance = compute_semantic_distance(real_image, fake_image)  # Real and fake images are Pillow Images

    if verbose:
        print(f"LPIPS (L): {lpips} \t PSNR (H): {psnr}")

    return  psnr, lpips, semantic_distance


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
