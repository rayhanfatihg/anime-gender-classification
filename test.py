import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from tqdm import tqdm
import os
from dataloader import get_dataloaders, AnimeDataset, data_transforms

# Configuration
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_model.pth'
OUTPUT_FILE = 'submission.csv'

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    # We use 'test' transform. 
    # We can rely on get_dataloaders or build dataset manually to be sure about order.
    # Let's build manually to ensure we strictly follow test.csv order.
    
    test_df = pd.read_csv('dataset/test.csv')
    print(f"Loading {len(test_df)} test samples...")
    
    test_dataset = AnimeDataset(
        df=test_df, 
        root_dir='dataset/test_images', 
        transform=data_transforms['test']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    # 2. Load Model structure
    print("Loading model...")
    weights = models.EfficientNet_V2_M_Weights.DEFAULT
    model = models.efficientnet_v2_m(weights=weights)
    
    # Re-apply the head modification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    # Load state dict
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded.")
    else:
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Train the model first!")
    
    model = model.to(DEVICE)
    model.eval()
    
    # 3. Inference
    predictions = []
    
    print("Starting inference...")
    with torch.no_grad():
        for inputs in tqdm(test_loader):
            inputs = inputs.to(DEVICE)
            
            # The dataset might return (image, label) or just image depending on dataloader.py
            # Our AnimeDataset returns (image, label) if columns > 1.
            # test.csv has 'id' only? Let's check logic.
            # Step 52: test.csv has "id" only (and maybe other columns if they were there).
            # Wait, our AnimeDataset checks `if len(self.df.columns) > 1`.
            # Step 52 showing test.csv content: line 2 is `img_...jpg`... wait, Line 1 is `id`.
            # It seems test.csv ONLY has 'id'. 
            # BUT submission_format.csv had `id,label`.
            # Let's assume test.csv only has 'id' based on previous cat.
            # So AnimeDataset will return ONLY image.
            
            # If AnimeDataset returns tuple (image, label), we unpack. If just image, we don't.
            # Let's handle both cases just to be safe.
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = inputs[0] # Assume image is first
            
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            
    # 4. Generate Submission
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission saved to {OUTPUT_FILE}")
    print(submission_df.head())

if __name__ == '__main__':
    main()
