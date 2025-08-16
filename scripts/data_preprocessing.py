import json, os, random
from pathlib import Path
from typing import Dict, Tuple, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import RandAugment, RandomErasing
from torchvision.utils import make_grid, save_image

# ---- config ----
DATA_ROOT = Path("data")
SPLITS = {"train":"train", "val":"val", "test":"test"}  # NOTE: using 'val'
CLASS_MAP_FILE = DATA_ROOT/"metadata"/"classes.json"
IMAGE_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 2)

VALID_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

class FolderDataset(Dataset):
    def __init__(self, root: Path, label_to_index: Dict[str,int], split: str, train: bool):
        self.root = root
        self.label_to_index = label_to_index
        self.train = train
        self.items: List[tuple] = []
        for cls in label_to_index.keys():
            cdir = root/cls
            if not cdir.exists(): 
                continue
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    self.items.append((p, label_to_index[cls]))

        if train:
            self.tf = T.Compose([
                T.Resize(int(IMAGE_SIZE*1.15)),
                T.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                RandAugment(num_ops=2, magnitude=7),               # stronger aug
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3), inplace=True)
            ])
        else:
            self.tf = T.Compose([
                T.Resize(IMAGE_SIZE + 32),
                T.CenterCrop(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        x = self.tf(im)
        return x, y

def make_loaders(batch_size=BATCH_SIZE):
    with open(CLASS_MAP_FILE) as f:
        label_to_index = json.load(f)["label_to_index"]

    loaders = {}
    for split, folder in SPLITS.items():
        split_dir = DATA_ROOT/folder
        train_flag = (split == "train")
        ds = FolderDataset(split_dir, label_to_index, split, train=train_flag)

        # --- class-balanced sampling for train ---
        if train_flag and len(ds) > 0:
            labels = [lbl for _, lbl in ds.items]
            # bincount needs at least size of num classes
            import numpy as np
            counts = np.bincount(labels, minlength=len(label_to_index))
            class_w = 1.0 / np.maximum(counts, 1)
            sample_w = [float(class_w[y]) for y in labels]
            sampler = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)
            shuffle = False
            drop_last = True
        else:
            sampler = None
            shuffle = False
            drop_last = False

        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=drop_last
        )
    return loaders.get("train"), loaders.get("val"), loaders.get("test")

def save_sample_grid(n=32, out="results/predictions/sample_augs.jpg"):
    os.makedirs(Path(out).parent, exist_ok=True)
    train_loader, _, _ = make_loaders(batch_size=n)
    xb, yb = next(iter(train_loader))
    grid = make_grid(xb[:n], nrow=8, normalize=True, value_range=(-2, 2))
    save_image(grid, out)
    print(f"Saved sample augment grid at {out}")

if __name__ == "__main__":
    save_sample_grid()
