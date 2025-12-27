import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tfms
from transformers import SiglipModel, CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, 
    LMSDiscreteScheduler, DDPMScheduler
)
from PIL import Image

from attention_pooler import AttentionPooler


class ViLexEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # vit
        self.vit = SiglipModel.from_pretrained("google/siglip-so400m-patch14-224").vision_model
        
        # attention pooling module
        self.attn_pooler = AttentionPooler(
            num_queries=75,
            embed_dim=768,
            patch_embed_dim=1152,
            num_heads=16,
            num_layers=5,
        )
        
    def forward(self, gt_rgb):
        vit_out = self.vit(gt_rgb)
        return self.attn_pooler(vit_out.last_hidden_state)


class SDModel(nn.Module):
    def __init__(self):
        super().__init__()

        # text encoder
        #tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        # diffusion model
        self.vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
        #self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.scheduler = DDPMScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")

    def forward(self, vilex_embs, noisy_latents, timesteps):
        clip_emb = self.text_encoder.text_model.embeddings(inputs_embeds=vilex_embs)
        clip_emb = self.text_encoder.text_model.encoder(clip_emb).last_hidden_state
        clip_emb = self.text_encoder.text_model.final_layer_norm(clip_emb)
        return self.unet(noisy_latents, timesteps, clip_emb)


class ViLexPipeline(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.encoder = ViLexEncoder()
        self.generator = SDModel()
        
        self.clip_bos_id = 49406
        self.clip_eos_id = 49407
        self.bos_emb = self.generator.text_encoder.text_model.embeddings.token_embedding.weight[self.clip_bos_id].to(device)
        self.eos_emb = self.generator.text_encoder.text_model.embeddings.token_embedding.weight[self.clip_eos_id].to(device)

    def generate(self, texts, noise_latent, timestep):
        pass

    def forward(self, gt_rgb, noisy_latent, timestep):
        vilex_embs = self.encoder(gt_rgb) # (B, 75, 768)
        # Add BOS and EOS tokens to the vilex embeddings
        vilex_embs = torch.cat([
            self.bos_emb.unsqueeze(0).unsqueeze(0).expand(vilex_embs.size(0), -1, -1),
            vilex_embs,
            self.eos_emb.unsqueeze(0).unsqueeze(0).expand(vilex_embs.size(0), -1, -1),
        ], dim=1)  # (B, 77, 768)
        generated_latent = self.generator(vilex_embs, noisy_latent, timestep)
        return generated_latent
