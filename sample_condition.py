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

import numpy as np
import PIL.Image as Image

import pandas as pd


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
    parser.add_argument('--save_dir', type=str, default='./results_semantic_norm2_anneal_10x_n_guid_images_4')
    parser.add_argument('--n_data_samples', type=int, default=1)
    parser.add_argument('--n_paths', type=int, default=1)
    parser.add_argument('--resample_every_steps', type=int, default=10)
    parser.add_argument('--potential_type', type=str, default='curr')
    parser.add_argument('--rs_temp', type=float, default=0.1)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--path_start_idx', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--anneal_scale', type=float, default=10)
    parser.add_argument('--anneal_amp', type=float, default=1)
    parser.add_argument('--anneal_loc', type=float, default=0.5)
    parser.add_argument('--kernel_idx', type=int, default=0)
    parser.add_argument('--guid_image_idx', type=int, default=3)
    parser.add_argument('--n_guid_images', type=int, default=1)
    parser.add_argument('--sem_scale', type=float, default=0.1)
    parser.add_argument('--measurement_scale', type=float, default=0.3)
    parser.add_argument('--anneal_factor', type=float, default=10.0)
    parser.add_argument('--record', action='store_true')
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
    np.random.seed(args.kernel_idx)  # Set random seed for kernel reproducibility
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_config['params'['sem_guid_scale']] = args.sem_scale
    cond_config['params'['scale']] = args.measurement_scale
    cond_config['params'['anneal_factor']] = args.anneal_factor
    cond_config['params'['do_anneal']] = args.do_anneal

    # Load the guidance image
    guid_images = []
    for i in range(args.n_guid_images):
        guid_filename = str(args.guid_image_idx + i).zfill(4)
        guid_image = Image.open(f'../guidance_images/{guid_filename}.png')
        guid_images.append(guid_image)

    # Pass the guidance image
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, guid_images=guid_images, **cond_config['params'])
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
                        rs_temp=args.rs_temp,
                        anneal_scale=args.anneal_scale,
                        anneal_loc=args.anneal_loc,
                        anneal_amp=args.anneal_amp,)

    # Directory name
    scale_factor = measure_config['operator'].get('scale_factor', 1)
    print('scale_factor = ', scale_factor)

    # if cond_config['method'] == 'ps_anneal':
    #     dir_name = f"{measure_config['operator']['name']}_noise_sigma_{measure_config['noise']['sigma']}_dps_anneal_amp_{args.anneal_amp}"
    # else:

    dir_name = f"{measure_config['operator']['name']}_semantic_{cond_config['params']['sem_guid_scale']}_measurement_{cond_config['params']['scale']}"
   
    # Working directory
    out_path = os.path.join(args.save_dir, dir_name)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon_paths', 'label', 'progress']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)

    

    subset = Subset(dataset, np.arange(args.start_idx, args.start_idx + args.n_data_samples, 1))
    loader = get_dataloader(subset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    # Save the kernel in case of motion blur
    if measure_config['operator']['name'] == 'motion_blur':
        kernel = operator.get_kernel()
        print(kernel)
        os.makedirs('../motion_blur_kernels_ood', exist_ok=True)
        kname = str(args.kernel_idx).zfill(5)
       
        plt.imsave(os.path.join('../motion_blur_kernels_ood', f'{kname}.png'), clear_color(kernel))

    # pathwise_psnr = np.zeros([args.n_data_samples, args.n_paths])
    # pathwise_lpips = np.zeros([args.n_data_samples, args.n_paths])
    # pathwise_distances = np.zeros([args.n_data_samples, args.n_paths])
        
    # Do Inference
    for img_idx, ref_img in enumerate(loader):
        logger.info(f"Inference for image {args.start_idx + img_idx}")
        fname = str(args.start_idx + img_idx).zfill(5)
        ref_img = ref_img.to(device)

        os.makedirs(os.path.join(out_path, 'recon_paths', f'{fname}'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'recon_paths_y', f'{fname}'), exist_ok=True)

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

        true_diff = y_n-y
        true_diff_reshaped = true_diff.reshape(true_diff.shape[0], -1)
        true_norm = torch.linalg.norm(true_diff_reshaped, dim=-1)

        print('true_norm = ', true_norm)
         
        # Sampling
        C, H, W = ref_img.shape[1], ref_img.shape[2], ref_img.shape[3]

        plt.imsave(os.path.join(out_path, 'input', fname + '.png'), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname + '.png'), clear_color(ref_img))

        for path_group_idx in range(args.n_paths // args.batch_size):
            path_curr_group_idx = args.path_start_idx + path_group_idx * args.batch_size
            if args.record:
                os.makedirs(os.path.join(out_path, 'progress', f'path#{path_curr_group_idx + 1}'), exist_ok=True)
            x_start = torch.randn((args.batch_size, C, H, W), device=device).requires_grad_()
            sample, measurement_distance, semantic_distance = sample_fn(x_start=x_start, measurement=y_n, record=args.record, save_root=out_path, path_curr_group_idx=path_curr_group_idx)

            y_space = operator.forward(sample)

            from compute_metrics import compute_psnr, compute_lpips

            for sample_idx in range(len(sample)):

                path_idx = args.path_start_idx + path_group_idx * args.batch_size + sample_idx

                psnr = compute_psnr(ref_img, sample[sample_idx].unsqueeze(0)).cpu().numpy()
                lpips = compute_lpips(ref_img, sample[sample_idx].unsqueeze(0)).cpu().numpy()
                logger.info(f"Path#{path_idx} | Method:{diffusion_config['sampler']} / PSNR: {psnr} / LPIPS: {lpips}")

                plt.imsave(os.path.join(out_path, 'recon_paths', f'{fname}', f'path#{path_idx}' + '.png'), clear_color(sample[sample_idx].unsqueeze(0)))
                plt.imsave(os.path.join(out_path, 'recon_paths_y', f'{fname}', f'path#{path_idx}_y_space' + '.png'), clear_color(y_space[sample_idx].unsqueeze(0)))

                curr_measurement_distance = measurement_distance[sample_idx].detach().cpu().numpy()
                curr_semantic_distance = semantic_distance[sample_idx].detach().cpu().numpy()

                print('curr_measurement_distance = ', curr_measurement_distance)
                print('curr_semantic_distance = ', curr_semantic_distance)

                # Dataframe
                df = pd.DataFrame({'Image': [args.start_idx + img_idx], \
                                   'Semantic Guidance Scale': [cond_config['params']['sem_guid_scale']], \
                                   'Measurement Guidance Scale': [cond_config['params']['scale']], \
                                   'Path': [path_idx], 'PSNR': [psnr], \
                                   'LPIPS': [lpips], \
                                   'Measurement Distance': [curr_measurement_distance], \
                                   'Semantic Distance': [curr_semantic_distance]})
                df.to_csv(os.path.join(args.save_dir, f'metrics.csv'), mode='a', header=True, index=True)

        print('true_norm = ', true_norm)



if __name__ == '__main__':
    main()