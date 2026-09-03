import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class DogsCatsDataset(Dataset):
    """
    PyTorch Dataset for Dogs and Cats Breed Classification.
    """
    def __init__(self,
                 root_dir: str | Path,
                 class_to_idx: Dict[str, int],
                 transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist.")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for class_name, class_idx in self.class_to_idx.items():
            class_folder = self.root_dir / class_name
            if class_folder.is_dir():
                for file_path in class_folder.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                        self.samples.append((file_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms(target_size: Tuple[int, int] = (224, 224)) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and validation/test transforms.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, eval_transform


def build_class_mapping(train_dir: str | Path, save_path: Optional[str | Path] = None) -> Dict[str, int]:
    """
    Scan train directory for class folders, sort them alphabetically, and build mapping.
    """
    train_path = Path(train_dir)
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    return class_to_idx


def compute_class_weights(dataset: DogsCatsDataset, num_classes: int) -> torch.Tensor:
    """
    Compute balanced class weights to address class imbalance.
    weight = total_samples / (num_classes * count_per_class)
    """
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    
    counts = np.clip(counts, a_min=1.0, a_max=None)
    total_samples = len(dataset)
    weights = total_samples / (num_classes * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(data_dir: str | Path = "data",
                    batch_size: int = 32,
                    num_workers: int = 2,
                    mapping_save_path: Optional[str | Path] = "models/class_mapping.json") -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create Train, Validation, and Test DataLoaders.
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    valid_dir = data_path / "valid"
    test_dir = data_path / "test"

    class_to_idx = build_class_mapping(train_dir, save_path=mapping_save_path)
    train_transform, eval_transform = get_transforms()

    train_dataset = DogsCatsDataset(train_dir, class_to_idx=class_to_idx, transform=train_transform)
    valid_dataset = DogsCatsDataset(valid_dir, class_to_idx=class_to_idx, transform=eval_transform)
    test_dataset = DogsCatsDataset(test_dir, class_to_idx=class_to_idx, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, valid_loader, test_loader, class_to_idx
