import torch
from detectron2.modeling import VisionMambaDet
import pdb

embed_dim, depth, num_heads, dp = 768, 12, 24, 0.1
model = VisionMambaDet(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        drop_path_rate=dp,
        pretrained="/lpai/ARM/mamba_mlp.pth",
        out_feature="last_feat",
        last_layer_process="add",
        bimamba_type="v3",
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        pt_hw_seq_len=14,
        if_cls_token=False,
    )

model = model.cuda()

inputs = torch.randn(2, 3, 1024, 1024)
inputs = inputs.cuda()

out = model(inputs)

print(out)