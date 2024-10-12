import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def eval_images(val_dataloader, model, device="cpu", ret_preds=False):
    model.eval()
    preds = []
    targets = []

    year_gallery = model.year_gallery.to(device)

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            # Get predictions (probabilities for each year based on similarity)
            logits_per_image = model(imgs, year_gallery)
            probs = logits_per_image.softmax(dim=-1)
            
            # Predict building year with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            
            # Store predicted indices and target answers
            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    year_gallery = year_gallery.cpu().numpy()
    preds = np.array(year_gallery[preds])[:,0]
    targets = targets[:,0]
    
    MAE_loss = np.mean(np.abs(targets - preds))
    
    model.train()

    if ret_preds:
        return MAE_loss, preds
    
    return MAE_loss
