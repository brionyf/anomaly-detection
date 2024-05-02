"""Loss function for Reverse Distillation."""

# Original Code
# Copyright (c) 2022 hq-deng
# https://github.com/hq-deng/RD4AD
# SPDX-License-Identifier: MIT

import torch
from torch import Tensor, nn

class ReverseDistillationLoss(nn.Module):
    """Loss function for Reverse Distillation."""

    def forward(self, encoder_features: list[Tensor], decoder_features: list[Tensor]) -> Tensor:
        """Computes cosine similarity loss based on features from encoder and decoder.

        Args:
            encoder_features (list[Tensor]): List of features extracted from encoder
            decoder_features (list[Tensor]): List of features extracted from decoder

        Returns:
            Tensor: Cosine similarity loss
        """
        cos_loss = torch.nn.CosineSimilarity()
        losses = list(map(cos_loss, encoder_features, decoder_features))
        loss_sum = 0
        for loss in losses:
            loss_sum += torch.mean(1 - loss)  # mean of cosine distance
        return loss_sum
        
def loss_function(model_outputs):
    a, b = model_outputs
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(model_outputs):
    a, b = model_outputs
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss
