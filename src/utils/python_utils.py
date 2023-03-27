# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
import math
import torch
import numpy as np
import SimpleITK as sitk

"""
Commonly used basic utilities

"""
# Author:
#  based on torchbiomedision python_utils.AverageMeter
#  modification: stores all values and calculates the standard deviation


class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Taken from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Useful for monitoring current value and running average over iterations.
    """
    def __init__(self):
        self.reset()
        self.values = np.array([])
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = np.array([])
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values = np.append(self.values,val)
        self.std = np.std(self.values)


class AccuracyMeter(object):
    """
    Computes and stores the predicted labels and computes performance measures.
    Taken from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Useful for monitoring current value and running average over iterations.
    """
    def __init__(self, n_classes):

        self.n_classes = n_classes

        self.reset()
        self.values = np.array([])

        self.class_correct = self.n_classes * [0.]
        self.class_total = self.n_classes * [0.]

        self.acc = self.n_classes * [-1.]

    def reset(self):

        self.class_correct = self.n_classes * [0.]
        self.class_total = self.n_classes * [0.]

        self.acc = self.n_classes * [-1.]

    def update(self, pred, gt):
        if np.isscalar(pred):
            pred = np.array(pred)
        if np.isscalar(gt):
            gt = np.array(gt)
        assert (len(pred) == len(gt)), 'input arguments must both be lists of same length or both scalars.'

        c = (pred == gt).squeeze()
        if len(c.shape) > 0:
            for i in range(len(c)):
                label = gt[i]
                self.class_correct[label] += c[i].item()
                self.class_total[label] += 1
        else:
            label = gt[0]
            self.class_correct[label] += c
            self.class_total[label] += 1

        for i in range(self.n_classes):
            self.acc[i] = 100 * self.class_correct[i] / max((self.class_total[i], 1))


def makeGaussian(dim, size, fwhm = [3,3], center=None):
    """ Make a square gaussian kernel.

    dim is the dimension of the gaussian: {2, 3}
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    if dim == 2:
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 / fwhm[0] ** 2 + (y - y0) ** 2 / fwhm[1] ** 2))
    elif dim == 3:
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        z = y[:, np.newaxis]

        for d in range(0,3):
            if center[d] is None:
                center[d] = size // 2

        return np.exp(-4 * np.log(2) * ((x - center[0]) ** 2 / fwhm[0] ** 2 + (y - center[1]) ** 2 / fwhm[1] ** 2 + (z - center[2]) ** 2 / fwhm[2] ** 2))


class OneHot(object):
    """
    Transform input tensor to a one-hot encoding along the channel dimension.
    """

    def __init__(self, n_labels):
        """

        Arguments
        ---------
        n_labels: int or tuple/list of ints
            number of labels to create one-hot encoding for
            The output nChannels will equal to n_labels.
        """
        self.n_labels = n_labels

    def __call__(self, *inputs):
        """
        Transform input tensor to a one-hot encoding along the channel dimension.
        Arguments
        ---------
        inputs : LongTensor or ByteTensor
            These will be converted to LongTensors before being used.

        Returns
        -------
        one hot encoded outputs
        """
        if not isinstance(self.n_labels, (tuple,list)):
            self.n_labels = [self.n_labels]*len(inputs)

        outputs = []
        for idx, _input in enumerate(inputs):
            in_size = tuple(_input.shape)
            out = torch.LongTensor(in_size[0], self.n_labels[idx], *in_size[2:])
            out.zero_()
            out.scatter_(1, _input, 1)
            # along dim 1 (1st arg), set 1 (3rd arg) at index given by _input (2nd arg)
            # The values along dim 1 of _input are the indices where 1's are to be set
            # in out along this same dim 1
            outputs.append(out)
        return outputs if idx > 0 else outputs[0]
    
def create_image_mask(in_image, closing=True, int_range=None):

    print('Image masks')

    mask = sitk.GetArrayFromImage(in_image)
    mask[np.where(mask > 1e-3)] = 1
    Mask = sitk.GetImageFromArray(mask)
    Mask.SetOrigin(in_image.GetOrigin())
    Mask.SetSpacing(in_image.GetSpacing())

    Mask = sitk.Cast(Mask, sitk.sitkUInt8)

    if closing:
        vectorRadius = (5, 5, 5)
        kernel = sitk.sitkBall
        Mask = sitk.BinaryMorphologicalClosing(Mask, vectorRadius, kernel)
    if int_range is not None:
        Mask = sitk.RescaleIntensity(Mask, int_range[0], int_range[1])

    for keys in in_image.GetMetaDataKeys():
        Mask.SetMetaData(keys, in_image.GetMetaData(keys))

    return Mask