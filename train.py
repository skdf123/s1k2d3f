# train.py 

import torch
import torch.nn as nn
import torch.optim as optim
from spi_loss import structured_prior_injection_loss
from models.unet import DiffusionUNet
from OBS_Diffusion.models.extract_skeleton import SkeletonExtractor
from utils.schedule import get_beta_schedule
from configs import get_config  
from dataset import get_dataloader 

config = get_config("configs.yml")
dataloader = get_dataloader(config)

model = DiffusionUNet(config).to(config.device)
skeleton_extractor = SkeletonExtractor(model_path="weights/skel_model.pth", device=config.device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

betas = get_beta_schedule(
    beta_schedule="linear", beta_start=1e-4, beta_end=0.02, num_diffusion_timesteps=1000
)

try:
    for epoch in range(config.n_epochs):
        for step, x in enumerate(dataloader):
            x = x.to(config.device)
            x = 2 * x - 1.0  # normalize to [-1, 1]

            noise = torch.randn_like(x[:, 3:])
            t = torch.randint(low=0, high=1000, size=(x.size(0),)).to(x.device)

            loss, loss_ddpm, loss_spi = structured_prior_injection_loss(
                model, x, t, noise, betas, epoch, config.n_epochs, skeleton_extractor
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"[Epoch {epoch}] Total: {loss.item():.4f} | DDPM: {loss_ddpm.item():.4f} | SPI: {loss_spi.item():.6f}")

except Exception as e:
    print(f"[ERROR] Training crashed: {e}")
