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
    parser.add_argument('--save_dir', type=str, default='./results_search')
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


    parser.add_argument('--ref_image_idxs', type=str, default='4')
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
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
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
    if cond_config['method'] == 'ps_anneal':
        dir_name = f"{measure_config['operator']['name']}_noise_sigma_{measure_config['noise']['sigma']}_dps_anneal_amp_{args.anneal_amp}"
    else:
        dir_name = f"{measure_config['operator']['name']}_noise_sigma_{measure_config['noise']['sigma']}_dps_scale_{cond_config['params']['scale']}"
   
    # Working directory
    out_path = os.path.join(args.save_dir, dir_name)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon_paths', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)

    subset = Subset(dataset, [int(idx)+1 for idx in args.ref_image_idxs.split(',')])
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
        os.makedirs('../motion_blur_kernels_exp', exist_ok=True)
        kname = str(args.kernel_idx).zfill(5)
       
        plt.imsave(os.path.join('../motion_blur_kernels_exp', f'{kname}.png'), clear_color(kernel))

    # pathwise_psnr = np.zeros([args.n_data_samples, args.n_paths])
    # pathwise_lpips = np.zeros([args.n_data_samples, args.n_paths])
    # pathwise_distances = np.zeros([args.n_data_samples, args.n_paths])
        
    # Do Inference
    for img_idx, ref_img in enumerate(loader):
        logger.info(f"Inference for image {args.start_idx + img_idx}")
        fname = args.ref_image_idxs[img_idx].zfill(5)
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
            x_start = torch.randn((args.batch_size, C, H, W), device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

            y_space = operator.forward(sample)

            from compute_metrics import compute_psnr, compute_lpips

            for sample_idx in range(len(sample)):

                path_idx = args.path_start_idx + path_group_idx * args.batch_size + sample_idx

                psnr = compute_psnr(ref_img, sample[sample_idx].unsqueeze(0))
                lpips = compute_lpips(ref_img, sample[sample_idx].unsqueeze(0))
                logger.info(f"Path#{path_idx + 1} | Method:{diffusion_config['sampler']} / PSNR: {psnr} / LPIPS: {lpips}")

                plt.imsave(os.path.join(out_path, 'recon_paths', f'{fname}', f'path#{path_idx + 1}' + '.png'), clear_color(sample[sample_idx].unsqueeze(0)))
                plt.imsave(os.path.join(out_path, 'recon_paths_y', f'{fname}', f'path#{path_idx + 1}_y_space' + '.png'), clear_color(y_space[sample_idx].unsqueeze(0)))

                # # Add title and save the best sample
                # plt.imshow(clear_color(sample[sample_idx].unsqueeze(0)))
                # plt.title(f"Path#{path_idx + 1} | PSNR: {psnr:.4f} LPIPS: {lpips:.4f} Distance: {sample_distance[sample_idx]:.4f}")
                # plt.axis('off')
                # # Save the plt
                # plt.savefig(os.path.join(out_path, f'recon_paths/{fname}', f'path#{path_idx + 1}' + '.png'))
                # plt.close()

                # pathwise_distances[args.start_idx + img_idx, path_idx] = sample_distance[sample_idx]
                # pathwise_psnr[args.start_idx + img_idx, path_idx] = psnr
                # pathwise_lpips[args.start_idx + img_idx, path_idx] = lpips

                # # Open the file quantitative results and write the results
                # with open(os.path.join(out_path, f'{fname}_pathwise.txt'), 'a') as f:
                #     f.write(f"Path#{path_idx}, PSNR: {psnr}, LPIPS: {lpips}, Distance: {sample_distance[sample_idx]} \n")

        # avg_psnr /= args.n_data_samples
        # avg_lpips /= args.n_data_samples

        # logger.info(f"Average PSNR: {avg_psnr} / Average LPIPS: {avg_lpips}")

        # # Open the file quantitative results and write the results
        # with open(os.path.join(out_path, 'quantitative_results.txt'), 'w') as f:
        #     f.write(f"Average PSNR: {avg_psnr}, Average LPIPS: {avg_lpips}")

        # Compute the norm of the true image with observation

        # print('true_norm = ', true_norm)

        # # np.save(os.path.join(out_path, 'pathwise_distances.npy'), pathwise_distances)
        # # np.save(os.path.join(out_path, 'pathwise_psnr.npy'), pathwise_psnr)
        # # np.save(os.path.join(out_path, 'pathwise_lpips.npy'), pathwise_lpips)

        # plt.close()
        # steps = np.arange(diffusion_config['steps']-1, -1, -1)
        # for n in range(len(scales)):
        #     plt.plot(steps, scales[n], label=f'Path: {n+1}')
        # plt.ylabel('Guidance Scale')
        # plt.xlabel('Steps')
        # plt.legend()

        # if task_config['conditioning']['method'] == 'ps_anneal':
        #     np.save(os.path.join(out_path, 'dps_anneal_guidance_scales.npy'), scales)
        #     plt.savefig(os.path.join(out_path, 'dps_anneal_guidance_scales.png'))
        # else:
        #     np.save(os.path.join(out_path, 'dps_guidance_scales.npy'), scales)  # Save the guidance scales for the DPS
        #     plt.savefig(os.path.join(out_path, 'dps_guidance_scales.png'))
        

if __name__ == '__main__':
    main()