import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io, transform


class ECGDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = int(self.annotations.iloc[index, 1])

        if self.transforms:
            image = self.transforms(image)

        return image, y_label