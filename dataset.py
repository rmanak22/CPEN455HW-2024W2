import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize
from bidict import bidict
from tqdm import tqdm
import pandas as pd
import pdb

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
replicate_color_channel = lambda x : x.repeat(3,1,1)

my_bidict = bidict({'Class0': 0,
                    'Class1': 1,
                    'Class2': 2,
                    'Class3': 3})

class CPEN455Dataset(Dataset):
    def __init__(self, root_dir='./data', mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            mode (string): One of 'train', 'validation', or 'test' to read the corresponding CSV.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Build list of image paths and labels from CSV.
        csv_path = os.path.join(self.root_dir, mode + '.csv')
        df = pd.read_csv(csv_path, header=None, names=['path', 'label'])
        self.samples = [(os.path.join(self.root_dir, path), label) for path, label in df.itertuples(index=False, name=None)]

        # Preload images into memory for faster retrieval.
        self.cached_data = []
        print(f"Preloading {len(self.samples)} images for mode '{mode}'...")
        for img_path, label in tqdm(self.samples, desc="Loading images"):
            image = read_image(img_path).type(torch.float32) / 255.
            if image.shape[0] == 1:
                image = replicate_color_channel(image)
            if self.transform:
                image = self.transform(image)
            self.cached_data.append((image, label))

    def __len__(self):
        # Return number of cached images.
        return len(self.cached_data)

    def __getitem__(self, idx):
        image, label = self.cached_data[idx]
        if label in my_bidict.values():
            category_name = my_bidict.inverse[label]
        else:
            category_name = "Unknown"
        return image, category_name

    def get_all_images(self, label):
        return [image for image, cat in self.cached_data if cat == label]

def show_images(images, categories, mode: str):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, image in enumerate(images):
        axs[i].imshow(image.permute(1, 2, 0))
        axs[i].set_title(f"Category: {categories[i]}")
        axs[i].axis('off')
    plt.savefig(mode + '_test.png')
    plt.show()

if __name__ == '__main__':
    transform_32 = Compose([
        Resize((32, 32)),
        rescaling
    ])
    dataset_list = ['train', 'validation', 'test']

    for mode in dataset_list:
        print(f"Mode: {mode}")
        dataset = CPEN455Dataset(root_dir='./data', mode=mode, transform=transform_32)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        for images, categories in tqdm(data_loader, desc=f"Processing {mode}"):
            print(images.shape, categories)
            images = torch.round(rescaling_inv(images) * 255).type(torch.uint8)
            show_images(images, categories, mode)
            break  # Only process one batch for demonstration
