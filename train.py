import io
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfms
from transformers import get_scheduler
from PIL import Image
from vilex import ViLexPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
use_wandb = True

class ParquetDataset(Dataset):
    def __init__(self, dataframe, vae):
        super().__init__()
        self.dataframe = dataframe
        self.vae = vae.to(torch.bfloat16).to(device)
        # normalize before input to SigLIP
        self.clip_norm = tfms.Compose([
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        im = Image.open(io.BytesIO(self.dataframe.iloc[idx]["jpg"]))
        # image must be square and have 3 channels
        im = tfms.CenterCrop(min(im.size))(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        
        # 1. VAE Latent Target (4, 64, 64)
        with torch.no_grad():
            vae_input = tfms.ToTensor()(tfms.Resize((512, 512))(im)).unsqueeze(0).to(torch.bfloat16).to(device) * 2 - 1
            latent = self.vae.encode(vae_input).latent_dist.sample().squeeze(0) * 0.18215
        
        # 2. RGB Input Conditioning (3, 224, 224)
        rgb_im = tfms.Resize((224, 224))(im)
        rgb = self.clip_norm(rgb_im)
        return latent, rgb

if __name__ == "__main__":
    model = ViLexPipeline(device=device)
    
    # Freeze params
    model.requires_grad_(False)
    model.encoder.attn_pooler.requires_grad_(True)
    #model.encoder.vit.requires_grad_(True)
    
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    optim_groups = [
        # {
        #     "params": [p for p in model.encoder.vit.parameters() if p.requires_grad],
        #     "lr": 1e-5,
        #     "weight_decay": 0.01,
        # }, 
        {
            "params": [p for p in model.encoder.attn_pooler.parameters() if p.requires_grad],
            "lr": 1e-4, # 3e-4
            "weight_decay": 0.01,
        },
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
    
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=1000, num_training_steps=300000) # 10k warmup

    # Data Loading
    df_list = [pd.read_parquet(f"data/{f}") for f in os.listdir("data/") if f.endswith(".parquet")]
    df = pd.concat(df_list, ignore_index=True)
    del df_list
    train_dataset = ParquetDataset(vae=model.generator.vae, dataframe=df)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(f"train: {len(train_loader)} batches")

    model.to(device)
    if use_wandb: wandb.init(project="vilex")
    
    model.train()
    step = 0
    ema_loss = None
    
    while True:
        for batch in train_loader:
            gt_latent, gt_rgb = batch
            gt_latent = gt_latent.to(torch.bfloat16).to(device)
            gt_rgb = gt_rgb.to(torch.bfloat16).to(device)

            timesteps = torch.randint(0, 1000, (gt_latent.size(0),), device=device).long()
            noise = torch.randn_like(gt_latent)
            noisy_latent = model.generator.scheduler.add_noise(gt_latent, noise, timesteps)

            alphas_cumprod = model.generator.scheduler.alphas_cumprod.to(device)
            snr_t = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
            snr_t = snr_t.clamp(max=5.0)

            with autocast(device_type=device, dtype=torch.bfloat16):
                model_pred = model(gt_rgb, noisy_latent, timesteps).sample

            # use loss Weighted by SNR per sample
            loss_elementwise = F.mse_loss(model_pred.float(), noise.float(), reduction='none')
            loss_per_sample = loss_elementwise.mean(dim=[1, 2, 3])
            snr_weights = snr_t / (snr_t + 1.0)
            loss = (loss_per_sample * snr_weights).mean()

            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = 0.99 * ema_loss + 0.01 * loss.item()

            # Logging ...
            current_lr = scheduler.get_last_lr()[0]
            if step % 100 == 0:
                print(f"step {step} | loss: {loss:.4f} | ema loss: {ema_loss:.4f} | timestep: {timesteps.item()}")
                if use_wandb:
                    wandb.log({
                        "train_loss": loss,
                        "ema_loss": ema_loss,
                        "grad_norm": grad_norm,
                        "lr": current_lr,
                        "step": step,
                    })

            if step % 5000 == 0 and step > 0:
                torch.save(model.encoder.state_dict(), "vilex_enc.pt")
            
            step += 1