# https://github.com/Ttl/ComfyUi_NNLatentUpscale/blob/master/nn_upscale.py

import torch
from comfy import model_management
import os
from .arch.dat_arch import DAT
from .arch.craft_arch import CRAFT
from .arch.swinfir_arch import SwinFIR
from .arch.drct_arch import drct
import wget


class SudoLatentUpscale:
    """
    Upscales SD1.5 latent using neural network
    Currently only working with fp32 and 2x scale
    """

    def __init__(self):
        self.local_dir = os.path.dirname(os.path.realpath(__file__))
        self.path = "models/"
        self.base_url = "https://github.com/styler00dollar/ComfyUI-sudo-latent-upscale/releases/download/models/"

        self.dtype = torch.float32
        self.weight_path = {
            "SwinFIR4x6_mse_1.5": os.path.join(
                self.local_dir, self.path, "SwinFIR4x6_mse_200k_1.5.pth"
            ),
            "CRAFT7x6_l1_eV2-b0_1.5": os.path.join(
                self.local_dir, self.path, "CRAFT7x6_l1_eV2-b0_150k_1.5.pth"
            ),
            "DAT6x6_l1_eV2-b0_1.5": os.path.join(
                self.local_dir, self.path, "DAT6x6_l1_eV2-b0_265k_1.5.pth"
            ),
            "DAT12x6_l1_eV2-b0_contextual_1.5": os.path.join(
                self.local_dir, self.path, "DAT12x6_l1_eV2-b0_contextual_315k_1.5.pth"
            ),
            "SwinFIR4x6_mse_xl": os.path.join(
                self.local_dir, self.path, "SwinFIR4x6_mse_64k_sdxl.pth"
            ),
            "SwinFIR4x6_fft_l1_xl": os.path.join(
                self.local_dir, self.path, "SwinFIR4x6_fft_l1_94k_sdxl.pth"
            ),
            "DRCT-l_12x6_325k_l1_xl": os.path.join(
                self.local_dir, self.path, "DRCT-l_12x6_325k_l1_sdxl.pth"
            ),
            "DRCTFIR-l_12x6_215k_l1_xl": os.path.join(
                self.local_dir, self.path, "DRCTFIR-l_12x6_215k_l1_sdxl.pth"
            ),
        }
        self.version = "none"

    def check_and_download(self, file_path: str):
        models_path = os.path.join(self.local_dir, self.path)
        if not os.path.exists(models_path):
            os.mkdir(models_path)

        if not os.path.exists(file_path):
            model_name = os.path.basename(file_path)
            url = self.base_url + model_name
            print("downloading: " + model_name)
            print("file_path: ", file_path)
            wget.download(url, out=file_path)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "version": (
                    [
                        "SwinFIR4x6_mse_1.5",
                        "CRAFT7x6_l1_eV2-b0_1.5",
                        "DAT6x6_l1_eV2-b0_1.5",
                        "DAT12x6_l1_eV2-b0_contextual_1.5",
                        "SwinFIR4x6_mse_xl",
                        "SwinFIR4x6_fft_l1_xl",
                        "DRCT-l_12x6_325k_l1_xl",
                        "DRCTFIR-l_12x6_215k_l1_xl"
                    ],
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, latent, version):
        device = model_management.get_torch_device()
        samples = latent["samples"].to(device=device, dtype=self.dtype)

        if version != self.version:
            self.check_and_download(self.weight_path[version])

            state_dict = torch.load(self.weight_path[version])

            # 24.4 M
            if version == "DAT6x6_l1_eV2-b0_1.5":
                self.model = DAT(
                    img_size=64,
                    in_chans=4,
                    embed_dim=270,
                    split_size=[8, 16],
                    depth=[6, 6, 6, 6, 6, 6],  # 6x6
                    num_heads=[6, 6, 6, 6, 6, 6],
                    expansion_factor=2,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    drop_path_rate=0.1,
                    use_chk=False,
                    upscale=2,
                    img_range=1.0,
                    resi_connection="1conv",
                    upsampler="pixelshuffle",
                )

            # 47.9 M
            if version == "DAT12x6_l1_eV2-b0_contextual_1.5":
                self.model = DAT(
                    img_size=64,
                    in_chans=4,
                    embed_dim=270,
                    split_size=[8, 16],
                    depth=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # 12x6
                    num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                    expansion_factor=2,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    drop_path_rate=0.1,
                    use_chk=False,
                    upscale=2,
                    img_range=1.0,
                    resi_connection="1conv",
                    upsampler="pixelshuffle",
                )

            # 46.1M
            if version == "CRAFT7x6_l1_eV2-b0_1.5":
                self.model = CRAFT(
                    in_chans=4,
                    embed_dim=180,
                    depths=[6, 6, 6, 6, 6, 6, 6],  # 7x6
                    num_heads=[6, 6, 6, 6, 6, 6, 6],
                    split_size_0=4,
                    split_size_1=16,
                    mlp_ratio=2.0,
                    qkv_bias=True,
                    qk_scale=None,
                    img_range=1.0,
                    upsampler="",
                    resi_connection="1conv",
                )

            # 3.8 M
            if version in ["SwinFIR4x6_mse_1.5", "SwinFIR4x6_mse_xl", "SwinFIR4x6_fft_l1_xl"]:
                self.model = SwinFIR(
                    patch_size=1,
                    in_chans=4,
                    embed_dim=96,
                    depths=[6, 6, 6, 6],
                    num_heads=[6, 6, 6, 6],
                    window_size=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    drop_path_rate=0.1,
                    ape=False,
                    patch_norm=True,
                    use_checkpoint=False,
                    upscale=2,
                    img_range=1.0,
                    upsampler="pixelshuffle",
                    resi_connection="SFB",
                )

            # 27.4M
            if version == "DRCT-l_12x6_325k_l1_xl":
                self.model = drct(
                in_chans=4,
                depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6), # 12x6
                num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                upscale=2,
                resi_connection="1conv"
            )

            # 27.9M
            if version == "DRCTFIR-l_12x6_215k_l1_xl":
                self.model = drct(
                in_chans=4,
                depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6), # 12x6
                num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                upscale=2,
                resi_connection="SFB"
            )

            self.model.load_state_dict(state_dict)
            self.model.to(self.dtype)

            self.version = version

        self.model.to(device=device).eval()
        with torch.inference_mode():
            latent_out = self.model(samples)

        latent_out = latent_out.to(device="cpu", dtype=self.dtype)
        self.model.to(device=model_management.vae_offload_device())
        return ({"samples": latent_out},)
