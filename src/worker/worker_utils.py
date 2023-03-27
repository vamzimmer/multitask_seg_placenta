import os
import torch
import shutil
import SimpleITK as sitk
from random import randint

class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, prefix='default_', model_name=None):
    filename = prefix + 'checkpoint.pth.tar'
    filename2 = prefix + 'model_best.pth.tar'
    filename3 = prefix + 'model_best_loss.pth.tar'

    if model_name is None:
        filename = "{}checkpoint.pth.tar".format(prefix)
        filename2 = "{}model_best.pth.tar".format(prefix)
        filename3 = "{}model_best_loss.pth.tar".format(prefix)
    else:
        filename = "{}checkpoint_{}.pth.tar".format(prefix, model_name)
        filename2 = "{}model_best_{}.pth.tar".format(prefix, model_name)
        filename3 = "{}model_best_loss_{}.pth.tar".format(prefix, model_name)
    torch.save(state, filename)

    if is_best == 1:
        print("Best validation performance")
        shutil.copyfile(filename, filename2)
    if is_best == 2:
        print("Best validation loss")
        shutil.copyfile(filename, filename3)


def get_image_to_show(image, plot_id=0):

    if len(image.size()) == 5:  # 3D: BxCxWxHxD
        sz = int(image.size()[-1] / 2)
        return image[plot_id, :, :, :, sz]
    elif len(image.size()) == 4:  # 2D: BxCxWxH
        return image[plot_id, :, :, :]
