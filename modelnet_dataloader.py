import os
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

def get_modelnet_loaders(
    root: str = "data/ModelNet40",
    batch_size: int = 32,
    num_points: int = 1024,
    use_augmentation: bool = True,
    train_set_size: int = None,
    balanced: bool = False
):
    """
    args:
        root (str): Root directory where the dataset will be stored/loaded.
        batch_size (int): Batch size for train and test loaders.
        num_points (int): Number of points to sample from the mesh for each shape.
        use_augmentation (bool): If True, applies random rotation, scaling, translation, 
                                 and jittering to the training dataset.
        train_set_size (int): Optional. Number of samples to use from the training set.
        balanced (bool): If True and train_set_size is set, sample an equal number of
                         instances per class. Requires train_set_size to be divisible by
                         the number of classes.

    returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # base transforms applied to all samples
    base_transforms = [
        T.SamplePoints(num_points), # samples a fixed number of points uniformly from the mesh faces
        T.NormalizeScale() # centers the point cloud and scales it to fit within a unit sphere
    ]
    
    # training augmentations if requested
    train_transforms = list(base_transforms)
    if use_augmentation:
        train_transforms.extend([
            T.RandomRotate(degrees=360, axis=2), # rotate around the Z-axis (gravity axis)
            T.RandomScale(scales=(0.8, 1.25)), # random scaling
            T.RandomJitter(translate=0.01) # random translation/jitter
        ])
        
    train_transform = T.Compose(train_transforms)
    test_transform = T.Compose(base_transforms)
    
    print(f"Loading/Downloading ModelNet40 train dataset to '{root}'...")
    train_dataset = ModelNet(
        root=root,
        name="40",
        train=True,
        transform=train_transform
    )
    
    if train_set_size is not None and train_set_size < len(train_dataset):
        if balanced:
            # Collect indices per class, then sample equally from each
            labels = np.array([train_dataset[i].y.item() for i in range(len(train_dataset))])
            unique_classes = np.unique(labels)
            num_classes = len(unique_classes)
            samples_per_class = train_set_size // num_classes
            
            if samples_per_class * num_classes != train_set_size:
                print(f"WARNING: train_set_size ({train_set_size}) is not evenly divisible "
                      f"by num_classes ({num_classes}). Using {samples_per_class} samples "
                      f"per class = {samples_per_class * num_classes} total samples.")
            
            selected_indices = []
            for cls in unique_classes:
                cls_indices = np.where(labels == cls)[0]
                perm = torch.randperm(len(cls_indices))[:samples_per_class]
                selected_indices.append(torch.tensor(cls_indices[perm.numpy()]))
            
            indices = torch.cat(selected_indices)
            print(f"Balanced sampling: {samples_per_class} samples/class × {num_classes} classes = {len(indices)} total")
        else:
            indices = torch.randperm(len(train_dataset))[:train_set_size]
        
        train_dataset = train_dataset[indices]
    
    print(f"Loading/Downloading ModelNet40 test dataset to '{root}'...")
    test_dataset = ModelNet(
        root=root,
        name="40",
        train=False,
        transform=test_transform
    )
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader

if __name__ == '__main__':
    print("Testing ModelNet40 DataLoader implementation...")

    batch_size = 4
    num_points = 1024
    
    try:
        train_loader, test_loader = get_modelnet_loaders(
            root="data/ModelNet40",
            batch_size=batch_size,
            num_points=num_points,
            use_augmentation=True
        )
        
        for batch in train_loader:
            print(f"  Batch object: {batch}")
            
            # batch.pos shape -> (batch_size * num_points, 3) 
            print(f"  Point coordinates (pos) shape: {batch.pos.shape}")
            print(f"  Labels (y) shape: {batch.y.shape}")
            print(f"  Batch vector (batch) shape: {batch.batch.shape}")
            
            # reshape into standard tensor format (batch, n, channels)
            pos_dense = batch.pos.view(batch_size, num_points, 3)
            y_dense = batch.y
            print(f"  Reshaped positions tensor shape: {pos_dense.shape} (Expected: [{batch_size}, {num_points}, 3])")
            print(f"  Labels tensor shape: {y_dense.shape} (Expected: [{batch_size}])")
            break
            
    except Exception as e:
        print("\nNote: Dataloader instantiated successfully, but running a batch failed.")
        print(f"Error: {e}")
        print("This is normal if the dataset has not yet been copied to 'data/ModelNet40/raw'.")
        print("Please copy the dataset (raw ModelNet40 files or zip) to 'data/ModelNet40' and try again.")
