import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import os

# Define transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]),
}

class AnimeDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (DataFrame): DataFrame containing image IDs and labels (optional).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except (IOError, FileNotFoundError):
             # Handle missing files gracefully, though ideally dataset should be clean
             # For now, just raise error to let user know
             raise FileNotFoundError(f"Image not found: {img_name}")

        if self.transform:
            image = self.transform(image)
            
        # Check if label exists (it might not for test set)
        if len(self.df.columns) > 1:
            label = int(self.df.iloc[idx, 1])
            return image, label
        else:
            return image

def get_dataloaders(batch_size=32, root_dir='.', val_split=0.2):
    """
    Creates and returns dataloaders for train, val, and test sets.
    
    Args:
        batch_size (int): Batch size for loading.
        root_dir (str): Root directory of the project (where 'dataset' folder is).
        val_split (float): Fraction of data to use for validation (default 0.2 for 80/20 split).
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    
    train_csv_path = os.path.join(root_dir, 'dataset/train.csv')
    test_csv_path = os.path.join(root_dir, 'dataset/test.csv')
    
    full_train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Split Train/Val
    train_df, val_df = train_test_split(full_train_df, test_size=val_split, random_state=42, stratify=full_train_df['label'])

    print(f"Training Info:")
    print(f"Total Labels: {len(full_train_df)}")
    print(f"Train Split: {len(train_df)}")
    print(f"Val Split: {len(val_df)}")
    print(f"Test Set: {len(test_df)}")

    # Create Datasets
    train_dir = os.path.join(root_dir, 'dataset/train_images')
    test_dir = os.path.join(root_dir, 'dataset/test_images') # Assuming val images are in train_images? 
    # Wait, the user said load image from @[dataset/train_images] and @[dataset/test_images]
    # Validation split comes from train.csv, so it should also use train_images. correct.
    
    train_dataset = AnimeDataset(df=train_df, root_dir=train_dir, transform=data_transforms['train'])
    val_dataset = AnimeDataset(df=val_df, root_dir=train_dir, transform=data_transforms['val'])
    test_dataset = AnimeDataset(df=test_df, root_dir=test_dir, transform=data_transforms['test'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = ['Masculine', 'Feminine'] 
    
    return train_loader, val_loader, test_loader, class_names
