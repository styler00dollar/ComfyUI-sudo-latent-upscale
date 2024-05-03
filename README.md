# ComfyUI-sudo-latent-upscale

This took heavy inspriration from [city96/SD-Latent-Upscaler](https://github.com/city96/SD-Latent-Upscaler) and [Ttl/ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale). Directly upscaling inside the latent space. 
Some models are for 1.5 and some models are for SDXL. All models are trained for drawn content. Might add new architectures or update models at some point. I recommend the SwinFIR or DRCT models.

1.5 comparison:
![comparison](https://github.com/styler00dollar/ComfyUI-sudo-latent-upscale/assets/51405565/9bae2125-9ffd-482c-aca5-023ab1e304b4)

SDXL comparison:
![comparison](https://github.com/styler00dollar/ComfyUI-sudo-latent-upscale/assets/51405565/bf86ad0b-442c-46c1-9ad0-cfb6d55a7963)

First row is upscaled rgb image from rgb models before being used in vae encode or vae decoded image for latent models, second row final output after second KSampler.

## Training Details
I tried to take promising networks from already existing papers and apply more exotic loss functions. The final [DAT](https://github.com/LeapLabTHU/DAT) model used a 4-channel 
[EfficientnetV2-b0](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py) as a discriminator, [Prodigy optimizer](https://pytorch-optimizers.readthedocs.io/en/latest/optimizer/#pytorch_optimizer.Prodigy)
with 0.1, bf16, L1 loss with a factor of 0.08 and batch size 32 for the normal model and 16 for the large model. Training takes around 22-24gb vram and was done on a 4090. I did not try 
vae decode inside training a lot, but if someone wants to train with it, use `torch.inference_mode()` to drastically reduce vram usage since vae does not get trained anyway. After training 
with a discriminator and l1, I applied [contextual loss](https://github.com/styler00dollar/Colab-traiNNer/blob/a8d97b4826c01d7b206e7a320156d8666db1efd2/code/loss/loss.py#L715) which used 
a self-made 4-channel latent classification network as a feature extractor. Training with contextual loss from scratch takes too long to converge, so I only used it at the very end. I can't really recommend the usage of contextual loss though.

Similar settings got applied to [CRAFT](https://github.com/AVC2-UESTC/CRAFT-SR) and I trained with batch size 16. I did not finetune CRAFT with contextual loss yet.

I then tried to train [SwinFIR](https://github.com/Zdafeng/SwinFIR) for 1.5. Prodigy with 1 and 0.1 caused massive instability, so I used Lamb with 3e-4, batch size 150, bf16 and MSE with 0.08. Final model was trained on 2x4090 with ddp and gloo, 100k steps each gpu.Prodigy with 1 and 0.1 caused massive instability, so I used Lamb with 3e-4, batch size 150, bf16 and MSE with 0.08. Final model was trained on 2x4090 with ddp and gloo, 100k steps each gpu.

For the SwinFIR SDXL models, I used Prodigy with 0.1, batch 140 and bf16. One model was trained with MSE and the other was trained with FFT and L1. Used weight 1 everywhere. Afterwards I trained some [DRCT](https://github.com/ming053l/DRCT) models with AdamW 1e-4, bf16, batch size 40 and 0.08 l1. Training DRCT took around 35gb vram.

### Further Ideas
Ideas I might test in the future:
- MSE
- Huber
- Different Conv2D (for example MBConv)
- Dropout prior to final conv
- Using normal image space as additional input or during training as loss with vae decode

### Failure cases
- 4 channel ssim on output latents. Only remains as positive loss if `nonnegative_ssim=True` gets set in `pytorch_msssim`, but model does not seem to train properly then.
- Using `vae.config.scaling_factor = 0.13025` (do not set a scaling factor, [nnlatent used it](https://github.com/Ttl/ComfyUi_NNLatentUpscale/blob/08105da31dbd7a54569661e135835e73bd8064b0/latent_resizer_train.py#L248) and city96 didn't, I do not recommend to use it), image range 0 to 1 (image tensor is supposed to be -1 to 1 prior to encoding with vae) and not using `torch.inference_mode()` while creating the dataset. A combination of these can make training a lot less stable, even if loss goes down during training and does seemingly converge, the final model won't be able to generate properly. Here is a correct example:
```python
vae = AutoencoderKL.from_single_file("vae.pt").to(device)
vae.eval()

with torch.inference_mode():
  image = cv2.imread(f)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = (
      torch.from_numpy(image.transpose(2, 0, 1))
      .float()
      .unsqueeze(0)
      .to(device)
      / 255.0
  )
  image = vae.encode(image*2.0-1.0).latent_dist.sample()
```
- [DITN](https://github.com/yongliuy/DITN) and [OmniSR](https://github.com/Francis0625/Omni-SR) looked like liquid with their official sizes. Not recommended to use small or efficient networks.

- [HAT](https://github.com/XPixelGroup/HAT) looked promising, but seemingly always had some kind of blur effect. I didn't manage to get a proper model yet.

![hat](https://github.com/styler00dollar/ComfyUI-sudo-latent-upscale/assets/51405565/d30ecd0a-b1be-4588-9155-7bddaa5d47bc)

- I tried to use [fourier](https://github.com/advimman/lama/blob/d4239d84c0e1040aabf7a95e3dc85cf728dfc9f4/saicinpainting/training/modules/ffc.py#L49) as first and last conv in DAT, but I didn't manage to properly train it yet. Making the loss converge seems hard.

![fourier](https://github.com/styler00dollar/ComfyUI-sudo-latent-upscale/assets/51405565/cabc7ad5-f577-45fc-9086-0fccd009b777)

- [GRL](https://github.com/ofsoundof/GRL-Image-Restoration) did not converge.
  
 ![grl](https://github.com/styler00dollar/ComfyUI-sudo-latent-upscale/assets/51405565/3ceba02f-2f0d-4872-8fc3-fc930785658a)

- SwinFIR with Prodigy 1 and Prodigy 0.1 caused massive instability. Images from my Prodigy 1, l1 and EfficientnetV2-b0 attempt.

![graphs](https://github.com/styler00dollar/ComfyUI-sudo-latent-upscale/assets/51405565/957aff0b-8670-471e-8f10-4231426a87c2)

![swinfir_prodigy](https://github.com/styler00dollar/ComfyUI-sudo-latent-upscale/assets/51405565/5c010b34-3b6e-4188-a561-e013bd52f185)

