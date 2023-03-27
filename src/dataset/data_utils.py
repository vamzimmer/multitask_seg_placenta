import numpy as np
import torch
import torchio as tio


def data_transforms(**kwargs):

    # default parameters and parameter parsing

    # intensity transformation
    whitening = kwargs["whitening"] if "whitening" in kwargs else False
    rescaling = kwargs["rescaling"] if "rescaling" in kwargs else [-1, 1]

    # resampling = kwargs["resample"] if "resample" in kwargs else 0
    croppad = kwargs["croppad"] if "croppad" in kwargs else True
    resample_size = kwargs["resample_size"] if "resample_size" in kwargs else [128,128,128]

    blur = kwargs["blur"] if "blur" in kwargs else False
    blur_prob = kwargs["blur_prob"] if "blur_prob" in kwargs else 0.25
    noise = kwargs["noise"] if "noise" in kwargs else False
    noise_prob = kwargs["noise_prob"] if "noise_prob" in kwargs else 0.25

    # spatial transformations
    spatial_prob = kwargs["spatial_prob"] if "spatial_prob" in kwargs else 0
    affine_prob = kwargs["affine_prob"] if "affine_prob" in kwargs else 0.5
    elastic_prob = kwargs["elastic_prob"] if "elastic_prob" in kwargs else 0.5

    flip_prob = kwargs["flip_prob"] if "flip_prob" in kwargs else 0
    flip_axes = kwargs["flip_axes"] if "flip_axes" in kwargs else ('LR','AP', 'IS')

    # construct intensity transformations
    tfms = []

    if whitening:
        rescale = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    else:
        rescale = tio.RescaleIntensity(rescaling, percentiles=(1, 99))
    
    # if resampling:
    if croppad:
        resample = tio.CropOrPad(resample_size)
    else:
        resample = tio.Resize(target_shape=resample_size)
    tfms.append(resample)

    if spatial_prob:
        spatial = tio.OneOf({                                # either
            tio.RandomAffine(image_interpolation='nearest'): affine_prob,               # random affine
            tio.RandomElasticDeformation(image_interpolation='nearest'): elastic_prob,   # or random elastic deformation
        }, p=spatial_prob)
        tfms.append(spatial)

    if flip_prob:
        flip = tio.RandomFlip(axes=flip_axes, flip_probability=flip_prob)
        tfms.append(flip)

    if blur:
        random_blur = tio.RandomBlur(p=blur_prob)
        tfms.append(random_blur)
    if noise:
        random_noise = tio.RandomNoise(p=noise_prob)
        tfms.append(random_noise)

    tfms.append(rescale)

    intensity_tfm = tio.Compose(tfms)
    validate_tfm = tio.Compose([resample, rescale])

    return intensity_tfm, validate_tfm


def create_tensor_from_torchio_subject(subject, image_names=('Image'), mask_names=('Label')):
    images = []
    masks = []
    image_count = 0
    mask_count = 0
    for idx, (image_name, image) in enumerate(subject.items()):

        if image_name in image_names:
            if image_count == 0:
                images = image.tensor
                image_count += 1
            else:
                images = torch.cat((images, image.tensor), 0)

        if image_name in mask_names:
            if mask_count == 0:
                masks = image.tensor
                mask_count += 1
            else:
                masks = torch.cat((masks, image.tensor), 0)

    return images, masks
