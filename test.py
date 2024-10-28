import torch
import models_pretrain
import utils.misc as misc



model = models_pretrain.__dict__["arm_base_pz16"](norm_pix_loss=True)
to_save = {'model': model.state_dict(), }

torch.save(to_save, "./mamba_mlp.pth")