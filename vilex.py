import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tfms
from transformers import SiglipModel, CLIPTextModel, CLIPTokenizer
from transformers.masking_utils import create_causal_mask
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

    def forward(self, vilex_embs, clip_attn_mask, noisy_latents, timesteps):
        # manually forward() the CLIPTextTransformer from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L533
        hidden_states = self.text_encoder.text_model.embeddings(inputs_embeds=vilex_embs)
        attention_mask = create_causal_mask(
            config=self.text_encoder.text_model.config,
            input_embeds=hidden_states,
            attention_mask=clip_attn_mask,
            cache_position=torch.arange(hidden_states.shape[1], device=hidden_states.device),
            past_key_values=None,
        )
        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=hidden_states, 
            attention_mask=attention_mask,
        )
        last_hidden_state = self.text_encoder.text_model.final_layer_norm(encoder_outputs.last_hidden_state)
        # forward UNet
        return self.unet(noisy_latents, timesteps, last_hidden_state)


class ViLexPipeline(nn.Module):
    def __init__(self, is_training, device="cuda"):
        super().__init__()
        self.encoder = ViLexEncoder()
        self.generator = SDModel()

        self.is_training = is_training

        self.clip_bos_id = self.generator.text_encoder.config.bos_token_id # 0
        self.clip_eos_id = self.generator.text_encoder.config.eos_token_id # 2
        self.bos_emb = self.generator.text_encoder.text_model.embeddings.token_embedding.weight[self.clip_bos_id].to(device)
        self.eos_emb = self.generator.text_encoder.text_model.embeddings.token_embedding.weight[self.clip_eos_id].to(device)

    def generate(self, texts, noise_latent, timestep):
        pass

    def forward(self, gt_rgb, noisy_latent, timestep):
        vilex_embs = self.encoder(gt_rgb) # (B, 75, 768)
        B, _, D = vilex_embs.size()

        # Add BOS and EOS tokens to the vilex embeddings
        vilex_embs = torch.cat([
            self.bos_emb.view(1, 1, -1).expand(B, -1, -1),
            vilex_embs,
            self.eos_emb.view(1, 1, -1).expand(B, -1, -1),
        ], dim=1)  # (B, 77, 768)
        _, T, _ = vilex_embs.size()

        # taildrop: randomly cover the last k tokens with eos
        if self.is_training:
            ks = torch.randint(2, T, (B,), device=vilex_embs.device)
            mask_bt = torch.arange(T, device=vilex_embs.device).unsqueeze(0) < ks.unsqueeze(1)
            mask_btd = mask_bt.unsqueeze(-1).expand(B, T, D)
            base = self.eos_emb.view(1, 1, D).expand(B, T, D)
            vilex_embs = torch.where(mask_btd, vilex_embs, base)
            #print(vilex_embs)

        generated_latent = self.generator(vilex_embs, mask_bt, noisy_latent, timestep)
        return generated_latent
