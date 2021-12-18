import torch

from catalyst import metrics
from torch import nn
from torch.nn import functional as F
from typing import OrderedDict as ORDType, Any

from gml_harm.model.backbones.resnet.utils import create_resnet


IMAGENET_MEAN = 255. * torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = 255. * torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class ResNetPL(nn.Module):
    def __init__(self, resnet_cfg: ORDType[str, Any]):
        super().__init__()
        self.impl = create_resnet(resnet_cfg)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, normed: bool = True):
        if not normed:
            pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
            target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        self.impl.to(pred)

        pred_feats = self.impl(pred)
        target_feats = self.impl(target)

        result = torch.stack([F.mse_loss(cur_pred, cur_target)
                              for cur_pred, cur_target
                              in zip(pred_feats, target_feats)]).sum()
        return result


class ResNetPLMetric(metrics.FunctionalBatchMetric):
    def __init__(self, metric_key, resnet_cfg: ORDType[str, Any], **kwargs):
        self.metric_fn = ResNetPL(resnet_cfg)
        super(ResNetPLMetric, self).__init__(
            metric_fn=self.metric_fn,
            metric_key=metric_key,
            **kwargs
        )