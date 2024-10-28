import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous

from .backbone import Backbone
from .utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
)

from .vit import SimpleFeaturePyramid
# add the root path to the system path
import sys, os
# import the parent directory of the current cwd
sys.path.append(os.path.dirname(os.getcwd()) + "/Finetuning")
from models_mamba import ARM
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
# from util import interpolate_pos_embed

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = torch.zeros((1, 1, embedding_size))
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            pos_tokens = pos_embed_checkpoint
        # B, N, C = pos_tokens.shape
        # new_pos_embed = torch.cat((pos_tokens[:, :N//2, :], extra_tokens, pos_tokens[:, N//2:, :]), dim=1)
        checkpoint_model['pos_embed'] = pos_tokens
        return checkpoint_model

logger = logging.getLogger(__name__)


__all__ = ["VisionMambaDet", "SimpleFeaturePyramid", "get_vim_lr_decay_rate"]


class VisionMambaDet(ARM, Backbone):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=24,
        use_checkpoint=False, 
        pretrained=None,
        if_fpn=True,
        last_layer_process="none",
        out_feature="last_feat",
        bimamba_type="v3",
        **kwargs,
    ):

        # for rope
        ft_seq_len = img_size // patch_size
        kwargs['ft_seq_len'] = ft_seq_len

        super().__init__(img_size, patch_size, depth=depth, embed_dim=embed_dim, channels=in_chans, num_classes=num_classes, bimamba_type=bimamba_type, **kwargs)

        self.use_checkpoint = use_checkpoint
        self.if_fpn = if_fpn
        self.last_layer_process = last_layer_process

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        # remove cls head
        del self.head
        del self.norm_f

        self.init_weights(pretrained)

        # drop the pos embed for class token
        if self.if_cls_token:
            del self.cls_token
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])
    

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = logging.getLogger(__name__)

            state_dict = torch.load(pretrained, map_location="cpu")
            state_dict_model = state_dict["model"]
            new_dict = {}
            for k, v in state_dict_model.items():
                if "conv1d" in k:
                    new_dict[k.replace("conv1d", "conv1d_b")] = v
                    new_dict[k.replace("conv1d", "conv1d_c")] = v
                    new_dict[k.replace("conv1d", "conv1d_c_b")] = v
                if "dt_proj" in k:
                    new_dict[k.replace("dt_proj", "dt_proj_b")] = v
                    new_dict[k.replace("dt_proj", "dt_proj_c")] = v
                    new_dict[k.replace("dt_proj", "dt_proj_c_b")] = v
                if "x_proj" in k:
                    new_dict[k.replace("x_proj", "x_proj_b")] = v
                    new_dict[k.replace("x_proj", "x_proj_c")] = v
                    new_dict[k.replace("x_proj", "x_proj_c_b")] = v
                if "A" in k:
                    new_dict[k.replace("A", "A_b")] = v
                    new_dict[k.replace("A", "A_c")] = v
                    new_dict[k.replace("A", "A_c_b")] = v
                if "D" in k:
                    new_dict[k.replace("D", "D_b")] = v
                    new_dict[k.replace("D", "D_c")] = v
                    new_dict[k.replace("D", "D_c_b")] = v
                new_dict[k] = v
            # new_dict.pop("head.weight")
            # new_dict.pop("head.bias")

            if self.patch_embed.patch_size[-1] != new_dict["patch_embed.proj.weight"].shape[-1]:
                new_dict.pop("patch_embed.proj.weight")
                new_dict.pop("patch_embed.proj.bias")
            interpolate_pos_embed(self, new_dict)

            res = self.load_state_dict(new_dict, strict=False) 
            logger.info(res)
            print(res)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def forward_features(self, x, inference_params=None):
        B, C, H, W = x.shape
        # x, (Hp, Wp) = self.patch_embed(x)
        x = self.patch_embed(x)

        batch_size, seq_len, _ = x.size()
        Hp = Wp = int(math.sqrt(seq_len))

        if self.pos_embed is not None:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        residual = None
        hidden_states = x
        features = []
        for layer in self.layers:
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
                if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)
                
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params, task= 'det'
                )


        residual = hidden_states
        # if self.last_layer_process == 'none':
        #     residual = hidden_states
        # elif self.last_layer_process == 'add':
        #     residual = hidden_states + residual
        # elif self.last_layer_process == 'add & norm':
        #     fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        #     residual = fused_add_norm_fn(
        #         hidden_states,
        #         self.norm_f.weight,
        #         self.norm_f.bias,
        #         eps=self.norm_f.eps,
        #         residual=residual,
        #         prenorm=False,
        #         residual_in_fp32=self.residual_in_fp32,
        #     )
 
        outputs = {self._out_features[0]: residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()}

        return outputs

    def forward(self, x):
        x = self.forward_features(x)
        return x


def get_vim_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=24):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".layers." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layers.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
