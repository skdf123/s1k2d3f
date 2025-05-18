# dataset.py(review-only)
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.img_names = sorted(os.listdir(self.input_dir))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        input_img = Image.open(os.path.join(self.input_dir, self.img_names[idx])).convert("RGB")
        target_img = Image.open(os.path.join(self.target_dir, self.img_names[idx])).convert("RGB")
        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)
        return torch.cat([input_tensor, target_tensor], dim=0)

def get_dataloader(config):
    dataset = SimpleDataset(config.data.train_data_dir)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=2)
    return dataloader, dataloader  # train_loader, val_loader
