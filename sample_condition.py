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
    parser.add_argument('--n_paths', type=int, default=1)
    parser.add_argument('--resample_every_steps', type=int, default=10)
    parser.add_argument('--potential_type', type=str, default='min')
    parser.add_argument('--rs_temp', type=float, default=0.1)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--no_guidance_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_path_groups', type=int, default=4)
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
                        no_guidance_steps=args.no_guidance_steps,
                        rs_temp=args.rs_temp)

    # Directory name
    dir_name = f"{measure_config['operator']['name']}"
   
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

    import numpy as np

    subset = Subset(dataset, np.arange(args.start_idx, args.start_idx + args.n_data_samples, 1))
    loader = get_dataloader(subset, batch_size=1, num_workers=0, train=False) 

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    pathwise_psnr = np.zeros([args.n_data_samples, args.n_paths])
    pathwise_lpips = np.zeros([args.n_data_samples, args.n_paths])
    pathwise_distances = np.zeros([args.n_data_samples, args.n_paths])
        
    # Do Inference
    for img_idx, ref_img in enumerate(loader):
        logger.info(f"Inference for image {img_idx}")
        fname = str(args.start_idx + img_idx).zfill(5)
        ref_img = ref_img.to(device)

        os.makedirs(os.path.join(out_path, 'recon_paths', f'{fname}'), exist_ok=True)

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

        plt.imsave(os.path.join(out_path, 'input', fname + '.png'), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname + '.png'), clear_color(ref_img))

        for path_group_idx in range(args.n_paths // args.batch_size):
            x_start = torch.randn((args.batch_size, C, H, W), device=device).requires_grad_()
            sample, sample_distance, sample_distances = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

            from compute_metrics import compute_psnr, compute_lpips

            for sample_idx in range(len(sample)):

                path_idx = path_group_idx * args.batch_size + sample_idx

                psnr = compute_psnr(ref_img, sample[sample_idx].unsqueeze(0))
                lpips = compute_lpips(ref_img, sample[sample_idx].unsqueeze(0))
                logger.info(f"Path#{path_idx + 1} | Method:{diffusion_config['sampler']} / PSNR: {psnr} / LPIPS: {lpips}")

                # Add title and save the best sample
                plt.imshow(clear_color(sample[sample_idx].unsqueeze(0)))
                plt.title(f"Path#{path_idx + 1} | PSNR: {psnr:.4f} LPIPS: {lpips:.4f} Distance: {sample_distance[sample_idx]:.4f}")
                plt.axis('off')
                # Save the plt
                plt.savefig(os.path.join(out_path, f'recon_paths/{fname}', f'path#{path_idx + 1}' + '.png'))
                plt.close()

                pathwise_distances[img_idx, path_idx] = sample_distance[sample_idx]
                pathwise_psnr[img_idx, path_idx] = psnr
                pathwise_lpips[img_idx, path_idx] = lpips

                # # Open the file quantitative results and write the results
                # with open(os.path.join(out_path, f'{fname}_pathwise.txt'), 'a') as f:
                #     f.write(f"Path#{path_idx}, PSNR: {psnr}, LPIPS: {lpips}, Distance: {sample_distance[sample_idx]} \n")

        # avg_psnr /= args.n_data_samples
        # avg_lpips /= args.n_data_samples

        # logger.info(f"Average PSNR: {avg_psnr} / Average LPIPS: {avg_lpips}")

        # # Open the file quantitative results and write the results
        # with open(os.path.join(out_path, 'quantitative_results.txt'), 'w') as f:
        #     f.write(f"Average PSNR: {avg_psnr}, Average LPIPS: {avg_lpips}")

    np.save(os.path.join(out_path, 'pathwise_distances.npy'), pathwise_distances)
    np.save(os.path.join(out_path, 'pathwise_psnr.npy'), pathwise_psnr)
    np.save(os.path.join(out_path, 'pathwise_lpips.npy'), pathwise_lpips)
        

if __name__ == '__main__':
    main()