import torch

from torch import nn

from gml_harm.model.idih.idih_entities import FeatureAggregation


class SimpleInputFusion(nn.Module):
    def __init__(self):
        super(SimpleInputFusion, self).__init__()
        self.feat_agg = FeatureAggregation(3, 1, mode='cat')
        self.block = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.BatchNorm2d(3)
        )

    def forward(self, a, b):
        c = self.feat_agg(a, b)
        return self.block(c)
