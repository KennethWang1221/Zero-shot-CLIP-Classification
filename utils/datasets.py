import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class CLIPDataset(Dataset):
    def __init__(self, image_path, image_filenames, captions, tokenizer, transforms, max_length):
        """
        image_filenames and captions must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.image_path = image_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.max_length = max_length
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=self.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{self.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=image)
        item['image'] = torch.tensor(transformed['image']).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)