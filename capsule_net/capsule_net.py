from typing import Tuple

import torch
import torch.nn as nn
from capsule_nn import CapsLinear

from .layers import PrimaryCapsule, TFLayer


class CapsuleNet(nn.Module):
    """Capsule Neural Network.

    Args:
        input_features: data size = [channels, length]
        classes: number of classes
        routings: number of routing iterations

    """

    def __init__(
        self,
        input_features: Tuple[int, int],
        classes: int,
        routings: int = 3,
        device=None,
    ):
        super().__init__()
        self.tf_feature_extractor = nn.Sequential(
            TFLayer(device=device),
            nn.BatchNorm1d(input_features[0], device=device),
            nn.Conv1d(input_features[0], 256, kernel_size=6, stride=2, device=device),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=6, stride=2, device=device),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=12, stride=4, device=device),
            nn.ReLU(),
        )
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(input_features[0], device=device),
            nn.Conv1d(input_features[0], 256, kernel_size=6, stride=2, device=device),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=6, stride=2, device=device),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=12, stride=4, device=device),
            nn.ReLU(),
        )
        self.primary_caps = PrimaryCapsule(
            256 * 2, 256, 8, kernel_size=6, stride=1, padding=0, device=device
        )
        # do one forward step to make sure the input shape of the digit capsule layer
        with torch.no_grad():
            feature1 = self.feature_extractor(
                torch.zeros(1, *input_features, device=device)
            )
            feature2 = self.tf_feature_extractor(
                torch.zeros(1, *input_features, device=device)
            )
            feature = torch.cat((feature1, feature2), dim=-2)
            digit_caps_in_shape = self.primary_caps(feature).shape[1:]


        # Capsule layer. Routing algorithm works here.
        self.digit_caps = CapsLinear(
            digit_caps_in_shape, (classes, 1), routings, device=device
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        fft_features = self.tf_feature_extractor(signal)
        features = self.feature_extractor(signal)
        features = torch.cat((fft_features, features), dim=-2)
        primary_caps_output = self.primary_caps(features)
        output = self.digit_caps(primary_caps_output)
        # the output is a vector, the length present the probability of the entity
        return output.norm(dim=-1)
