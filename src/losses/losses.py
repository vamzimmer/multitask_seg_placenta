"""

Losses taken from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
DiceLoss
DiceBCELoss
IoULoss
FocalLoss
TverskyLoss
FocalTverskyLoss

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """
    This loss combines Dice loss with the standard binary cross-entropy (BCE) loss that is generally the default for
    segmentation models. Combined the two methods allows for some diversity in the loss, while benefitting from the
    stability of BCE.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs) # was not commented for segm_exp4

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE