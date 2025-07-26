import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class DeepGlobeDataset(Dataset):
    """
    Base Dataset class for DeepGlobe Land Cover Classification.
    Handles loading of image and mask pairs.
    """
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.data_dir = os.path.join(root_dir, split)

        # List and sort all image files (assuming they end with '_sat.jpg')
        all_files = os.listdir(self.data_dir)
        self.images = sorted([f for f in all_files if f.endswith('_sat.jpg')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # Infer mask name from image name
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        image_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        # Open image as RGB, mask as grayscale ('L')
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # Load as grayscale for raw pixel values

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Ensure mask is a long tensor and squeeze any channel dimension if present
        # The target_transform will already convert to tensor, so just squeeze and cast.
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze(0).long()
        else: # In case target_transform didn't include ToTensor, do it here
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


class PatchedDeepGlobeDataset(DeepGlobeDataset):
    """
    Patched Dataset class for DeepGlobe Land Cover Classification.
    This version correctly maps RGB mask colors to integer class labels (0-6)
    and handles potential JPEG artifacts by binarizing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define the official RGB colors for each class based on dataset documentation
        self.colormap = np.array([
            [0, 0, 0],          # 0=Unknown (or background if not explicitly defined)
            [0, 255, 255],      # 1=Urban
            [255, 255, 0],      # 2=Agriculture
            [255, 0, 255],      # 3=Rangeland
            [0, 255, 0],        # 4=Forest
            [0, 0, 255],        # 5=Water
            [255, 255, 255]     # 6=Barren
        ])

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        image_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = Image.open(image_path).convert('RGB')
        mask_rgb = Image.open(mask_path).convert('RGB') # Load mask as RGB to use colormap

        # Apply image transformations (resize, augmentations, ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)
        # Apply mask transformations (resize). Note: ToTensor will be applied *after* color-to-label mapping.
        if self.target_transform:
            mask_rgb = self.target_transform(mask_rgb)

        # Convert PIL Image to NumPy array for pixel-wise comparison
        mask_rgb_np = np.array(mask_rgb)

        # Binarize the mask to remove JPEG compression artifacts.
        # Pixels that are "mostly" a color will become that color exactly (255).
        # This helps in precise color matching with the colormap.
        binarized_mask_np = (mask_rgb_np > 128).astype(np.uint8) * 255

        # Create an empty array to store the integer class labels (0-6)
        # It will have the same spatial dimensions as the mask.
        mask_labels = np.zeros((binarized_mask_np.shape[0], binarized_mask_np.shape[1]), dtype=np.uint8)

        # Iterate through the predefined colormap to map RGB colors to class indices
        for class_index, color in enumerate(self.colormap):
            # Find all pixels in the binarized mask that exactly match the current color
            matches = np.all(binarized_mask_np == color, axis=-1)
            mask_labels[matches] = class_index

        # Convert the final label map (NumPy array of integers) to a PyTorch LongTensor
        mask = torch.from_numpy(mask_labels).long()

        return image, mask
