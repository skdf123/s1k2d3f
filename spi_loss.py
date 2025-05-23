# spi_loss.py (review-only)

import torch
import torch.nn.functional as F

def compute_skeleton_loss(pred_skel, gt_skel):
    return F.mse_loss(pred_skel, gt_skel, reduction='none').mean(dim=(1, 2, 3))

def structured_prior_injection_loss(model, x0, t, noise, betas, epoch, max_epochs, skeleton_extractor, lambda_s=100):
    """
    SPI loss with time/epoch scheduling (scheduling functions simplified for review release).
    """

    device = x0.device
    alpha_prod = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    # Forward diffusion process
    noisy_input = x0[:, 3:] * alpha_prod.sqrt() + noise * (1.0 - alpha_prod).sqrt()

    # Predict noise
    model_input = torch.cat([x0[:, :3], noisy_input], dim=1)
    predicted_noise = model(model_input, t.float())

    # Reverse to predicted clean image
    x0_pred = (noisy_input - predicted_noise * (1 - alpha_prod).sqrt()) / alpha_prod.sqrt()

    # Skeleton maps
    skel_pred = skeleton_extractor.run_inference(x0_pred)
    skel_gt = skeleton_extractor.run_inference(x0[:, 3:])

    # Structure loss
    skel_loss = compute_skeleton_loss(skel_pred, skel_gt)

    # Scheduled SPI weights (simplified)
    # Total diffusion steps (default DDPM setting)
    T = 1000
    t_scaled = t.float() / T
    # Placeholder schedule (review-only): simplified linear mask.
    # Actual injection uses a dynamic schedule (see paper Sec 5.1, Eq. X)
    time_weight = 1.0 - t_scaled 
    epoch_weight = epoch / max_epochs  


    spi_loss = (time_weight * skel_loss).mean() * epoch_weight
    ddpm_loss = (noise - predicted_noise).square().sum(dim=(1, 2, 3)).mean()

    total_loss = ddpm_loss + lambda_s * spi_loss
    return total_loss, ddpm_loss, spi_loss


