import torch
import numpy as np
from PIL import Image
from torchvision import models,transforms
from torch.utils.data import Dataset

class ClothData(Dataset):
    def __init__(self, path, images_df):
        self.path = path
        self.images_df = images_df
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        image = Image.open(self.path + 'images/' + self.images_df['filename'][idx])
        # convert image to numpy array
        # image=np.asarray(image).astype(np.float32)
        labels = self.images_df.iloc[idx, 1:4].values.astype(np.long)

        label1 = torch.tensor(labels[0], dtype=torch.long)
        label2 = torch.tensor(labels[1], dtype=torch.long)
        label3 = torch.tensor(labels[2], dtype=torch.long)

        image = self.transform(image)
        return {
            'features': image, 'label1': label1, 'label2': label2, 'label3': label3
        }
