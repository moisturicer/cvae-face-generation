import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

# only picked 6 attributes that are visually obvious enough to see the difference when toggling them during the demo
ATTRIBUTES = ['Smiling', 'Young', 'Eyeglasses', 'Male', 'Bald', 'Heavy_Makeup']

# 64x64 is a good balance, small enough to train fast on colab, large enough to still see facial features clearly
IMG_SIZE = 64


def get_transforms():
    # normalizing to [-1, 1] because the decoder uses tanh which outputs in that range, they need to match
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_df, transform=None):
        self.img_dir = img_dir
        self.attr_df = attr_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        img_name = self.attr_df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # celeba stores attributes as -1 and 1 converting to 0 and 1 makes it easier to work with
        attrs = torch.tensor(
            self.attr_df.iloc[idx][ATTRIBUTES].values.astype(float),
            dtype=torch.float32
        )
        return img, attrs


def get_loaders(data_dir, batch_size=64):
    img_dir = os.path.join(data_dir, 'img_align_celeba/img_align_celeba')
    attr_path = os.path.join(data_dir, 'list_attr_celeba.csv')

    attr_df = pd.read_csv(attr_path)

    # convert -1/1 to 0/1
    attr_df[ATTRIBUTES] = (attr_df[ATTRIBUTES] == 1).astype(int)

    # celeba has a standard train/test split at 162000, following the official split so results are comparable
    train_df = attr_df.iloc[:162000]
    test_df = attr_df.iloc[162000:]

    transform = get_transforms()
    train_dataset = CelebADataset(img_dir, train_df, transform)
    test_dataset = CelebADataset(img_dir, test_df, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


def dataset_summary(data_dir):
    attr_path = os.path.join(data_dir, 'list_attr_celeba.csv')
    attr_df = pd.read_csv(attr_path)
    attr_df[ATTRIBUTES] = (attr_df[ATTRIBUTES] == 1).astype(int)

    print(f"total images: {len(attr_df)}")
    print(f"train split:  162000")
    print(f"test split:   {len(attr_df) - 162000}")
    print(f"\nattribute distribution (% positive):")
    for attr in ATTRIBUTES:
        pct = attr_df[attr].mean() * 100
        print(f"  {attr:20s}: {pct:.1f}%")