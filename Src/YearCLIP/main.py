import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yearclip import YearCLIP
from yearclip import train
from yearclip.train import eval_images
from yearclip.train.dataloader import YearDataLoader, img_train_transform, img_val_transform

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def save_model(model):
    folder = "yearclip/model/weights/"
    torch.save(model.image_encoder.mlp.state_dict(), folder+"image_encoder_mlp_weights.pth")
    torch.save(model.year_encoder.state_dict(),      folder+"year_encoder_weights.pth")
    torch.save(model.logit_scale,                    folder+"logit_scale_weights.pth")

if __name__ == "__main__":
    
    Epoch = 16
    batch_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YearCLIP(from_pretrained=False).to(device)
    
    optimizer = torch.optim.Adam([
        {"params": model.image_encoder.parameters(), "lr": 3e-4, "weight_decay": 1e-6},
        {"params": model.year_encoder.parameters(),  "lr": 3e-5, "weight_decay": 1e-6},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.87)
    
    train_data = YearDataLoader("../../Dataset/csv/train.csv", "../../Dataset/images", transform=img_train_transform())
    val_data   = YearDataLoader("../../Dataset/csv/valid.csv", "../../Dataset/images", transform=img_val_transform())
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4, drop_last=True)
    val_dataloader   = DataLoader(val_data  , batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    min_loss = float("inf")
    cur_loss = 0.0
    losses = []
    cnt = patience = 5
    for i in range(Epoch):
        train(train_dataloader, model, optimizer, i+1, batch_size, device, scheduler)
        cur_loss = eval_images(val_dataloader, model, device)
        losses.append(cur_loss)
        print(f"Validation MAE_Loss: {cur_loss}")
        
        cnt -= 1
        if cur_loss < min_loss:
            save_model(model)
            min_loss = cur_loss
            cnt = patience
        
        if cnt == 0: break
    
    print("Losses:", losses)