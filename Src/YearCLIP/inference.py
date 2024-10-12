import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yearclip import YearCLIP
from yearclip.train import eval_images
from yearclip.train.dataloader import YearDataLoader, img_val_transform

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == "__main__":
    
    batch_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YearCLIP().to(device)
    
    csv_path = "../../Dataset/csv/test.csv"
    test_data   = YearDataLoader(csv_path, "../../Dataset/images", transform=img_val_transform())
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    MAE_loss, preds = eval_images(test_dataloader, model, device, ret_preds=True)
    
    df = pd.read_csv(csv_path)
    names = df['Picture'].to_numpy()
    years = df['Year'].to_numpy()
    diffs = preds - years
    
    df = pd.DataFrame({
        'name': names,
        'year': years,
        'pred': preds,
        'diff': diffs
    })

    df.to_csv('../../Results/YearCLIPv1.csv',index=False)
    
    print("Test MAE_Loss:", MAE_loss)