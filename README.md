<img src=assets/vilex.JPG />

# open-vilex

An attempt to replicate the ViLex pipeline from the research paper [Visual Lexicon: Rich Image Features in Language Space](https://arxiv.org/abs/2412.06774) in PyTorch using open source components.

ViLex maps images directly into the text space, while preserving complex visual details that are difficult to express in natural language. The learned ViLex tokens act as a text-like "language" which can be used to reconstruct an image or combined with natural language in prompts.

Since the original work was built on the proprietary Imagen model, the diffusion model is replaced with [Stable Diffusion 1.5](https://github.com/CompVis/stable-diffusion). Imagen is a pixel-space diffusion model with cascaded upscalers but SD1.5 generates in the latent space, so there are additional VAE encoding/decoding steps and the resolution is different. Finally, the text encoder of SD1.5 is CLIP ViT-L/14 instead of OpenCLIP ViT-H/14. However, the setup is still compatible for the ViLex method.

## to do
- [x] Implement attention pooling module
- [x] Define vilex pipeline
- [x] Add train script
- [x] Implement TailDrop
- [ ] Add results from 90k training steps
- [ ] Add inference notebook to demo reconstructions with different amounts of vilex tokens
- [ ] Add inference notebook to demo combining vilex tokens with natural language tokens in prompts

## references
https://arxiv.org/abs/2412.06774