import copy
import os

import torch
from torch import nn

from networks.swinunet import SwinTransformerSys

__all__ = ["SwinUNet"]


class SwinUNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        in_channels=3,
        img_size=352,
        patch_size=4,
        embed_dim=96,
        depths=None,
        decoder_depths=None,
        num_heads=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_ckpt=None,
        require_pretrained=False,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        if decoder_depths is None:
            decoder_depths = [2, 2, 2, 1]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.img_size = img_size
        self.num_classes = num_classes
        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            depths_decoder=decoder_depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            final_upsample="expand_first",
        )

        if require_pretrained and not pretrained_ckpt:
            raise ValueError("SwinUNet training requires --pretrained_ckpt.")
        if pretrained_ckpt:
            self.load_pretrained(pretrained_ckpt)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.swin_unet(x)

    def load_pretrained(self, pretrained_path):
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

        print(f"SwinUNet pretrained_path: {pretrained_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=device)

        if "model" not in pretrained_dict:
            print("--- start load pretrained model by splitting ---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    del pretrained_dict[k]
            self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            return

        pretrained_dict = pretrained_dict["model"]
        print("--- start load pretrained model for encoder and decoder ---")

        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3 - int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k: v})

        for k in list(full_dict.keys()):
            if k in model_dict and full_dict[k].shape != model_dict[k].shape:
                print(f"delete:{k}; shape pretrain:{full_dict[k].shape}; shape model:{model_dict[k].shape}")
                del full_dict[k]

        self.swin_unet.load_state_dict(full_dict, strict=False)
