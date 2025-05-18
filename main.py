# main.py 

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from dataset import Data
from models import DenoisingDiffusion

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs.yml', type=str, help="Path to config file")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(args.local_rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config.local_rank = args.local_rank

    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)
        config.device = torch.device("cuda", config.local_rank)
    else:
        config.device = torch.device("cpu")


    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')


    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
        
    if dist.get_rank() == 0:
        print(f"[INFO] Launching SkelDiff training on device: {config.device}, rank: {config.local_rank}")


    dataset = Data(config)
    diffusion = DenoisingDiffusion(config)
    diffusion.train(dataset)

if __name__ == "__main__":
    main()
