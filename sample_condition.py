from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import Subset

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--n_data_samples', type=int, default=1)
    parser.add_argument('--n_particles', type=int, default=1)
    parser.add_argument('--resample_every_steps', type=int, default=10)
    parser.add_argument('--potential_type', type=str, default='min')
    parser.add_argument('--rs_temp', type=float, default=0.1)
    parser.add_argument('--l1', type=float, default=0.0)
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, l1=args.l1, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    logger.info(f"Sampling: {diffusion_config['sampler']} / Steps: {diffusion_config['steps']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, 
                        model=model, 
                        measurement_cond_fn=measurement_cond_fn, 
                        operator=operator, 
                        resample_every_steps=args.resample_every_steps,
                        potential_type=args.potential_type,
                        rs_temp=args.rs_temp)

    # Directory name
    dir_name = f"{measure_config['operator']['name']}_{diffusion_config['sampler']}_{diffusion_config['steps']}_steps_potential_{args.potential_type}_l1_{args.l1}"
   
    # Working directory
    out_path = os.path.join(args.save_dir, dir_name)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)

    subset = Subset(dataset, range(args.n_data_samples))
    loader = get_dataloader(subset, batch_size=1, num_workers=0, train=False) 


    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask, l1=args.l1)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        C, H, W = ref_img.shape[1], ref_img.shape[2], ref_img.shape[3]

        import numpy as np
        from PIL import Image


        y_n_im = (clear_color(y_n) * 255).astype(np.uint8)
        y_n_pil = Image.fromarray(y_n_im)
        y_n_pil.save(os.path.join(out_path, 'progress', f'input_{fname}'))

        # Resize to (256, 256)
        y_n_pil_resized = y_n_pil.resize((256, 256), Image.LANCZOS)
        y_n_pil_resized.save(os.path.join(out_path, 'progress', f'resized_{fname}'))

        x_start = torch.randn((args.n_particles, C, H, W), device=device).requires_grad_()
        sample, distance = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))

        for i in range(len(sample)):
            plt.imsave(os.path.join(out_path, 'recon', f'id_{i}_{fname}'), clear_color(sample[i].unsqueeze(0)))

        best_sample = sample[torch.argmin(distance)]
        plt.imsave(os.path.join(out_path, 'recon', f'best_recon_{fname}'), clear_color(best_sample.unsqueeze(0)))

        avg_sample = sample.mean(dim=0).unsqueeze(0)
        plt.imsave(os.path.join(out_path, 'recon', f'avg_recon_{fname}'), clear_color(avg_sample))


        # Compute PSNR and LPIPS

        from compute_metrics import compute_psnr, compute_lpips
        psnr = compute_psnr(ref_img, best_sample.unsqueeze(0))
        lpips = compute_lpips(ref_img, best_sample.unsqueeze(0))
        logger.info(f"Method:{diffusion_config['sampler']} / PSNR: {psnr} / LPIPS: {lpips}")


if __name__ == '__main__':
    main()
