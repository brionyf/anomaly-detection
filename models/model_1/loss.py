"""Loss function for Reverse Distillation."""

# Original Code
# Copyright (c) 2022 hq-deng
# https://github.com/hq-deng/RD4AD
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

# from .model_1 import calculate_anomaly_map

class ReverseDistillationLoss(nn.Module):
    """Loss function for Reverse Distillation."""

    def forward(self, batch):#, resize_ratio):
        normal_loss = self.normal_loss(batch) #self.normal_loss2(batch)
        anomaly_loss = self.anomaly_loss(batch)#, resize_ratio) #self.anomaly_loss2(batch, image_size)
        # print(">>>>>>>>>>>>>>>> Normal Loss: {} \tAnomaly Loss: {}".format(normal_loss, anomaly_loss))
        # return normal_loss + 0.3 * anomaly_loss
        return normal_loss, anomaly_loss

    def normal_loss(self, batch):
        encoder_features, decoder_features = batch['model_outputs']
        cos_loss = torch.nn.CosineSimilarity()
        losses = list(map(cos_loss, encoder_features, decoder_features))
        loss_sum = 0
        for loss in losses:
            loss_sum += torch.mean(1 - loss)  # mean of cosine distance
        return loss_sum

    def anomaly_loss(self, batch, mode="multiply"):
        encoder_features, decoder_features = batch['model_outputs']
        image_size = batch['image'].shape[2:] #[int(val / resize_ratio) for val in batch['image'].shape[2:]]
        # print(">>>>>>>>>>>>>>>> Loss: image size = {}".format(image_size))
        anomaly_maps = torch.ones([encoder_features[0].shape[0], 1, image_size[0], image_size[1]], device=encoder_features[0].device)  # b c h w
        if mode == "add":
            anomaly_maps *= 0
        for idx, (student_feature, teacher_feature) in enumerate(zip(encoder_features, decoder_features)):
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=image_size[0], mode="bilinear", align_corners=True)
            if mode == "multiply":
                anomaly_maps *= distance_map
            elif mode == "add":
                anomaly_maps += distance_map
        mask_batch = batch['mask']
        # print(">>>>>>>>>>>>>>>> anomaly maps = {} ({}, {}) \tmasks = {}".format(anomaly_maps.shape, anomaly_maps.min(), anomaly_maps.max(), mask_batch.shape))
        anomaly_loss = self.dice_loss(anomaly_maps, mask_batch, batch["image"])
        return anomaly_loss

    def normal_loss2(self, batch):
        cos_loss = torch.nn.CosineSimilarity()
        encoder_features, decoder_features = batch['model_outputs'][0]
        losses = list(map(cos_loss, encoder_features, decoder_features))
        # print(">>>>>>>>>>>>>>>> losses shape = {} ({})".format(len(losses), losses[0].shape))  # Result: 3 (torch.Size([8, 64, 64]))
        encoder_features, decoder_features = batch['model_outputs'][1]  # rotated image features
        for item in list(map(cos_loss, encoder_features, decoder_features)):
            losses.append(item)
        # print(">>>>>>>>>>>>>>>> appended losses shape = {} ({})".format(len(losses), losses[0].shape))  # Result: 6 (torch.Size([8, 64, 64]))
        loss_sum = 0
        for loss in losses:
            loss_sum += torch.mean(1 - loss)  # mean of cosine distance
        return loss_sum

    def anomaly_loss2(self, batch, image_size, mode="multiply"):
        encoder_features, decoder_features = batch['model_outputs'][0]
        rot_enc_feats, rot_dec_feats = batch['model_outputs'][1]
        anomaly_maps = get_anomaly_maps(encoder_features, decoder_features, image_size, mode)
        # print(">>>>>>>>>>>>>>>> anomaly maps = {} ({}, {})".format(anomaly_maps.shape, anomaly_maps.min(), anomaly_maps.max()))  # Result: torch.Size([8, 1, 256, 256]) (8.514294313499704e-05, 0.28607261180877686)
        rot_anomaly_maps = get_anomaly_maps(rot_enc_feats, rot_dec_feats, image_size, mode)
        anomaly_maps *= TF.rotate(rot_anomaly_maps, angle=-90)
        mask_batch = batch['mask']
        # print(">>>>>>>>>>>>>>>> anomaly maps = {} ({}, {}) \tmasks = {}".format(anomaly_maps.shape, anomaly_maps.min(), anomaly_maps.max(), mask_batch.shape))  # Result: torch.Size([8, 1, 256, 256]) (9.405534129314219e-09, 0.05139555037021637)
        anomaly_loss = self.dice_loss(anomaly_maps, mask_batch, batch["image"])
        return anomaly_loss

# Source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    def dice_loss(self, preds, targets, originals):
        # METHOD 1
        # preds = F.sigmoid(preds)
        numerator = 2 * torch.sum(preds * targets) # 2 * tf.reduce_sum(y_true * y_pred)
        denominator = torch.sum(preds + targets) # tf.reduce_sum(y_true + y_pred)
        dice = (numerator + 1) / (denominator + 1) # numerator / (denominator + tf.keras.backend.epsilon())

        # METHOD 2
        # preds = F.sigmoid(preds)
        # preds = preds.view(-1) # flatten tensors
        # targets = targets.view(-1)
        # smooth = 1.
        # intersection = (preds * targets).sum()
        # dice = (2. * intersection + smooth)/(preds.sum() + targets.sum() + smooth)

        return 1 - dice

def get_anomaly_maps(encoder_features, decoder_features, image_size, mode):
    anomaly_maps = torch.ones([encoder_features[0].shape[0], 1, image_size[0], image_size[1]], device=encoder_features[0].device)  # b c h w
    if mode == "add":
        anomaly_maps *= 0
    for idx, (student_feature, teacher_feature) in enumerate(zip(encoder_features, decoder_features)):
        distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
        distance_map = torch.unsqueeze(distance_map, dim=1)
        distance_map = F.interpolate(distance_map, size=image_size[0], mode="bilinear", align_corners=True)
        if mode == "multiply":
            anomaly_maps *= distance_map
        elif mode == "add":
            anomaly_maps += distance_map
    return anomaly_maps
