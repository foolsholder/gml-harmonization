import torch

from .simple_input_fusion import SimpleInputFusion


def create_mask_fusion(mask_fusion_cfg: str) -> torch.nn.Module:
    if mask_fusion_cfg == 'rgb':
        return SimpleInputFusion()
    elif mask_fusion_cfg == 'identity':
        return torch.nn.Identity()
    raise f"MaskFusion not implemented yet - {mask_fusion_cfg}"