import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from label_generator import generate_labels


## TODO: Make custom dataset for your purposes and rename this from CustomImageDataset to something cool
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_flag, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.DataFrame(generate_labels(img_dir, label_flag))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample