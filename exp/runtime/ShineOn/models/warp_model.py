import torch.nn as nn

from ShineOn.models.networks.cpvton.warp import (
    FeatureExtraction,
    FeatureL2Norm,
    FeatureCorrelation,
    FeatureRegression,
    TpsGridGen,
)

# coding=utf-8


class WarpModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.extractionA = FeatureExtraction(7,
            ngf=64,
            n_layers=3,
            norm_layer=nn.BatchNorm2d,
        )
        self.extractionB = FeatureExtraction(
            3, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d
        )
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(
            input_nc=192, output_dim=2 * 5 ** 2
        )
        self.gridGen = TpsGridGen(
            256, 192, grid_size=5
        )

    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)

        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        return grid, theta