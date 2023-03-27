import os
import sys
import numpy as np
import torch
import pandas as pd
from openpyxl import load_workbook
from glob import glob as glob
from argparse import ArgumentParser
from shutil import copyfile
import SimpleITK as sitk
import time

from src.fusion.polar import cart2pol_info, pol2cart
from src.utils.python_utils import makeGaussian

import src.utils.resample_utils as rutils
from src.utils.python_utils import create_image_mask


# Author: Veronika Zimmer <vam.zimmer@gmail.com, veronika.zimmer@tum.de>
# 	  King's College London, UK
#     TU Munich, Germany


def fuse_images(image_files, output_file, **varargin):
    """
    Image fusion of two or three ultrasound images.

    ---------
    Arguments:
    `image_files` : list of image file names
    `output_file` : output file name

    Optional arguments:
    `interp` : interpolation method {'linear', 'nearest', 'bspline'}, default: 'linear'
    `mask_files` : list of image mask file names, default: None
    `transform_files` : image transformations, default: None (default transformations are used)
    `flip` : string for flipping images, {'0 0', '1 1', '0 1', '0 0 0', '1 1 1'}, default: '0 1'
    `params` : extra alignment parameters, default: None
    `fusion_method` : method for image fusion, {'maximum', 'average', 'frustum', 'alignment'}, default: 'frustum'
    `weight` : weight type for frustum fusion, {'gauss', 'gauss+signal'}, default: 'gauss+signal'
    `sigma` : sigma of all dimensions for gaussian weight image
    `center` : center of gaussian in weight image
    `default_transform` : which default transforms to use. {1, 2}
                          1: Two transducer holder
                          2: Three transducer holder

    Usage: fuse_images(image_files, output_file, opt_arg1=x, opt_arg2=[y], ...)
    """

    # default varargin
    # for k, v in varargin.items():
    #     print('{}: {}'.format(k, v))
    if 'dimension' not in varargin:
        varargin['dimension'] = 3
    if 'interp' not in varargin:
        varargin['interp'] = 'linear'
    if 'mask_files' not in varargin:
        varargin['mask_files'] = None
    if 'output_mask_file' not in varargin:
        varargin['output_mask_file'] = None
    if 'transform_files' not in varargin:
        varargin['transform_files'] = None
    if 'flip' not in varargin:
        varargin['flip'] = [0, 1]
    if 'params' not in varargin:
        varargin['params'] = None
    # else:
    #     if os.path.isfile(varargin['params']):
    #         varargin['params'] = read_parameter_file(varargin['params'])
    #     else:
    #         varargin['params'] = None
    if 'fusion_method' not in varargin:
        varargin['fusion_method'] = 'frustum'
    if 'reference' not in varargin:
        varargin['reference'] = None
    if 'weight' not in varargin:
        varargin['weight'] = 'gauss+signal'
    if 'sigma' not in varargin:
        varargin['sigma'] = [160, 150, 500]
    if 'center' not in varargin:
        varargin['center'] = [100, None, None]
    if 'default_transform' not in varargin:
        varargin['default_transform'] = 1
    if 'weight_dir' not in varargin:
        varargin['weight_dir'] = None

    start = time.time()

    # noinspection SpellCheckingInspection
    nimages = len(image_files)
    assert nimages > 1, 'Please provide more than one image for the fusion.'
    # read images
    images = []
    for n in range(0, nimages):
        images.append(sitk.ReadImage(image_files[n]))
    # read masks
    masks = []
    set_masks = False
    if isinstance(varargin['mask_files'], list):
        # print("Read masks")
        assert len(varargin['mask_files']) == nimages, 'wrong number of mask files provided. ' \
                                           'Expected {:d}, got {:d}.'.format(nimages, len(varargin['mask_files']))
        for n in range(0, nimages):
            masks.append(sitk.ReadImage(varargin['mask_files'][n]))
        set_masks = True

    # Different fusion methods
    fused = 0
    fused_mask = 0
    if varargin['fusion_method'] == 'frustum':
        weighted_image_fusion = WeightedFrustumImageFusion(dimension=varargin['dimension'], number_images=nimages, interp=varargin['interp'])
        weighted_image_fusion.set_weights(varargin['weight'])
        if set_masks:
            weighted_image_fusion.set_masks(masks)
        weighted_image_fusion.set_sigma(varargin['sigma'])
        weighted_image_fusion.set_center(varargin['center'])
        weighted_image_fusion.set_default_transforms(varargin['default_transform'])
        fused = weighted_image_fusion(images, varargin['params'], varargin['flip'])
        fused_mask = weighted_image_fusion.get_fused_mask()
        if varargin['weight_dir'] is not None:
            weight_images = weighted_image_fusion.get_weight_images()
            if isinstance(weight_images, list):
                for n in range(0, nimages):
                    name = os.path.basename(image_files[n])
                    sitk.WriteImage(weight_images[n], varargin['weight_dir'] + '/' + os.path.splitext(name)[0] + str(
                        int(varargin['default_transform'])) + '.mha')
 
    elif varargin['fusion_method'] == 'maximum':
        maximum_image_fusion = MaximumImageFusion(dimension=varargin['dimension'], number_images=nimages, interp=varargin['interp'])
        if set_masks:
            maximum_image_fusion.set_masks(masks)
        maximum_image_fusion.set_default_transforms(varargin['default_transform'])
        if varargin['reference'] is not None:
            maximum_image_fusion.set_reference(varargin['reference'])
        fused = maximum_image_fusion(images, varargin['params'], varargin['flip'])
        fused_mask = maximum_image_fusion.get_fused_mask()
    elif varargin['fusion_method'] == 'average':
        average_image_fusion = AverageImageFusion(dimension=varargin['dimension'], number_images=nimages, interp=varargin['interp'])
        if set_masks:
            average_image_fusion.set_masks(masks)
        average_image_fusion.set_default_transforms(varargin['default_transform'])
        fused = average_image_fusion(images, varargin['params'], varargin['flip'])
        fused_mask = average_image_fusion.get_fused_mask()
    elif varargin['fusion_method'] == 'addition':
        addition_image_fusion = AdditionImageFusion(dimension=varargin['dimension'], number_images=nimages, interp=varargin['interp'])
        if set_masks:
            addition_image_fusion.set_masks(masks)
        addition_image_fusion.set_default_transforms(varargin['default_transform'])
        fused = addition_image_fusion(images, varargin['params'], varargin['flip'])
        fused_mask = addition_image_fusion.get_fused_mask()
    elif varargin['fusion_method'] == 'alignment':
        image_alignment = ImageFusion(dimension=varargin['dimension'], number_images=nimages, interp=varargin['interp'])
        if set_masks:
            image_alignment.set_masks(masks)
        image_alignment.set_default_transforms(varargin['default_transform'])
        aligned_images = image_alignment(images, varargin['params'], varargin['flip'])
        fused_mask = image_alignment.get_fused_mask()

        out_dir = os.path.dirname(output_file)
        for n in range(0, nimages):
            name = os.path.basename(image_files[n])
            sitk.WriteImage(aligned_images[n], out_dir + '/' + os.path.splitext(name)[0] + str(int(varargin['default_transform'])) + '.mhd')
            
            if varargin['output_mask_file'] is not None:
                mask_dir = os.path.dirname(varargin['output_mask_file'])
                sitk.WriteImage(fused_mask[n + 1], mask_dir + '/' + os.path.splitext(name)[0] + str(int(varargin['default_transform'])) + '.mha')

    if str(fused.__class__) == "<class 'SimpleITK.SimpleITK.Image'>":
        sitk.WriteImage(fused, output_file)
    if varargin['output_mask_file'] is not None and str(fused_mask.__class__) == "<class 'SimpleITK.SimpleITK.Image'>":
        sitk.WriteImage(fused_mask, varargin['output_mask_file'])

    end = time.time()
    print("Elapsed time for {} fusion of {} images: {}".format(varargin['fusion_method'], nimages, end - start))


class ImageFusion(object):

    def __init__(self, dimension=3, number_images=2, interp='linear'):
        """
        Base class for the fusion of two or three ultrasound images.

        Arguments
        ---------
        `dimension` : image dimension
        `number_images` : number of images to fuse
        `interp` : string or list/tuple of string
            possible values from this set: {'linear', 'nearest', 'bspline'}
            Different types of interpolation can be provided for each input,
            e.g. for two inputs, `interp=['linear','nearest']

        """
        if dimension == 2 or dimension == 3:
            self._dimension = dimension
        else:
            assert False, "wrong image dimension. Only 2, 3 are supported."
        self._number_images = number_images
        self._interp = interp

        self._transforms = self._number_images * [sitk.Transform(self._dimension, sitk.sitkIdentity)]
        self._default_transforms = 1

        self._new_origin = None
        self._new_size = None
        self._new_spacing = None

        self._image_masks = [None] * self._number_images
        self._fused_mask = None
        self._set_masks = False

        self._reference = None

    def set_reference(self, ref):
        reference = sitk.ReadImage(ref)
        self._reference = reference

    """
        Set/Get functions.
    """
    def set_transforms(self, transforms):
        """
        Aligning of images before the fusion.
        """
        assert len(transforms) == self._number_images

        self._default_transforms = False
        for i in range(0,self._number_images):
            self._transforms[i] = sitk.ReadTransform(transforms[i])

    def set_default_transforms(self, arg):
        assert arg in range(0, 5), 'wrong value {} for default transforms. Only [0, 1, 2, 3, 4] are supported.'.format(arg)

        self._default_transforms = arg

    def get_default_transform(self):
        return self._default_transforms

    def set_masks(self, arg):
        assert isinstance(arg, list) and len(arg)==self._number_images, 'wrong argument type or wrong format.'
        self._image_masks = arg
        self._set_masks = True

    def get_masks(self):
        if self._set_masks is True:
            return self._image_masks
        else:
            return None

    def get_fused_mask(self):
        return self._fused_mask

    """
        Call
    """
    def __call__(self, images, params=None, flip=None):

        # check input
        assert len(images) == self._number_images, \
            'wrong number of images. Got {}, expected {}.'.format(len(images), self._number_images)
        for i in range(0, self._number_images):
            assert isinstance(images[i], sitk.SimpleITK.Image), 'input {} not an image!'.format(i)
        if flip is not None:
            assert isinstance(flip, list), 'wrong format of flip. List expected.'
            assert len(flip) == self._number_images, \
                'wrong flip format. Got {} parameters, expected {}.'.format(len(flip), self._number_images)
        else:
            flip = np.zeros(self._number_images)

        if params is not None:
            params = np.fromstring(params, dtype=float, sep=' ')
            assert len(params) == self._number_images * self._dimension,  \
                'wrong parameter format. Got {} parameters, expected {}.'.format(len(params), self._number_images * self._dimension)
        else:
            params = np.zeros(self._dimension*self._number_images)
        params = params.reshape(self._dimension, self._number_images)

        if self._default_transforms:
            if  self._default_transforms == 1:
                assert self._number_images == 2, 'wrong number of images for default transform type {}. ' \
                                               'Got {}, expected 2.'.format(self._default_transforms, self._number_images)
            elif self._default_transforms == 2:
                assert self._number_images == 3, 'wrong number of images for default transform type {}. ' \
                                               'Got {}, expected 3.'.format(self._default_transforms, self._number_images)

        # prepare transforms
        if self._default_transforms:
            if self._default_transforms == 1:
                self._prepare_transforms2(images, params, flip)
            elif self._default_transforms == 2:
                self._prepare_transforms3(images, params, flip)

        else:
            self._change_transforms(images, params, flip)

        fused, aligned = self._fuse(images)

        return fused

    """
        This function has to be overwritten by classes inheriting from ImageFusion!
    """
    def _fuse(self, images):
        """
            In the base class, the aligned images are given.
        """

        # create image masks
        if self._set_masks is False:
            for n in range(0, self._number_images):
                self._image_masks[n] = create_image_mask(images[n])

        # get new image bounds
        if self._reference is None:
            self._get_new_bounds(self._image_masks)
        else:
            self._new_origin = self._reference.GetOrigin()
            self._new_spacing = self._reference.GetSpacing()
            self._new_size = self._reference.GetSize()

        transformed = self._prepare_images(images)
        transformed_masks = self._prepare_images(self._image_masks, interp='nearest')

        # compute images masks
        combined = sitk.GetArrayFromImage(transformed_masks[0])
        for n in range(1, self._number_images):
            combined = combined + sitk.GetArrayFromImage(transformed_masks[n])
        self._fused_mask = sitk.GetImageFromArray(combined)
        self._fused_mask.SetOrigin(transformed[0].GetOrigin())
        self._fused_mask.SetSpacing(transformed[0].GetSpacing())

        return transformed, transformed_masks

    """
        Internal functions.
    """
    def _prepare_images(self, images, interp=None):
        """
            Aligning of images before the fusion.
        """

        outputs = []

        if interp is None:
            interp = self._interp

        for i in range(0, self._number_images):
            transformed = rutils.image_resample(images[i], transform=self._transforms[i], out_origin=self._new_origin,
                                                 out_size=self._new_size, out_spacing=self._new_spacing, interp=interp)
            transformed = self._set_metadata(transformed, images[i])
            outputs.append(transformed)

        return outputs

    @staticmethod
    def _set_metadata(image, reference):
        """
            Copying of image meta data from reference to image.
        """

        keys = reference.GetMetaDataKeys()
        for k in range(0, len(keys)):
            image.SetMetaData(keys[k], reference.GetMetaData(keys[k]))

        return image

    def _change_transforms(self, images, params, flip):
        """
            Modify transforms: Add transform for image flipping and for additional alignment parameters.
        """

        for i in range(0, self._number_images):
            if flip[i]:
                flipping = sitk.AffineTransform(self._dimension)
                center = np.asarray(images[i].GetOrigin()) + np.multiply(np.asarray(images[i].GetSpacing()), np.asarray(images[i].GetSize())) / 2
                if images[i].GetSize()[2] == 1:
                    center[2] = 0
                flipping.SetCenter(center)
                matrix = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
                flipping.SetMatrix(matrix.ravel())
            else:
                flipping = sitk.Transform(self._dimension, sitk.sitkIdentity)

            affine = sitk.AffineTransform(self._dimension)
            center = np.asarray(images[i].GetOrigin()) + np.multiply(np.asarray(images[i].GetSpacing()),np.asarray(images[i].GetSize())) / 2
            affine.SetCenter(center)

            angle = params[0, i]
            matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            affine.SetMatrix(matrix.ravel())

            if self._dimension == 2:
                affine.SetTranslation((params[1, i], 0))
            elif self._dimension == 3:
                affine.SetTranslation((params[1, i], params[2, i], 0))

            composite = sitk.Transform(self._dimension, sitk.sitkComposite)
            composite.AddTransform(flipping)
            # composite.AddTransform(self._transforms[i])
            composite.AddTransform(affine)
            self._transforms[i] = composite

    def _prepare_transforms2(self, images, params, flip):
        """
            Default transform for the alignment of two images.
        """

        # composite1 = sitk.Transform(self._dimension, sitk.sitkComposite)
        composite1 = sitk.CompositeTransform(self._dimension)

        if flip[0]:
            flipping = sitk.AffineTransform(self._dimension)
            center = np.asarray(images[0].GetOrigin()) + np.multiply(np.asarray(images[0].GetSpacing()),
                                                                     np.asarray(images[0].GetSize())) / 2
            if images[0].GetSize()[2] == 1:
                center[2] = 0
            flipping.SetCenter(center)
            matrix = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
            flipping.SetMatrix(matrix.ravel())

            composite1.AddTransform(flipping)

        euler11 = sitk.Euler3DTransform()
        euler11.SetParameters((0, 0, 0.2617993877991494, 40, 0, 0))
        euler11.SetFixedParameters((0, -15.169497244391824, 0, 0))
        euler12 = sitk.Euler3DTransform()
        euler12.SetParameters((0, 0, 0 + params[0, 0], 0 + params[1, 0], 0 + params[2, 0], 0))
        euler12.SetFixedParameters((-18.45 + params[1, 0], 79.72 + params[2, 0], 0, 0))

        composite1.AddTransform(euler11)
        composite1.AddTransform(euler12)
        self._transforms[0] = composite1

        # ------------------------------------------------------------------------------------------------------------

        # composite2 = sitk.Transform(self._dimension, sitk.sitkComposite)
        composite2 = sitk.CompositeTransform(self._dimension)

        if flip[1]:
            flipping = sitk.AffineTransform(self._dimension)
            center = np.asarray(images[1].GetOrigin()) + np.multiply(np.asarray(images[1].GetSpacing()), np.asarray(images[1].GetSize())) / 2
            if images[1].GetSize()[2] == 1:
                center[2] = 0
            flipping.SetCenter(center)
            matrix = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
            flipping.SetMatrix(matrix.ravel())

            composite2.AddTransform(flipping)

        euler21 = sitk.Euler3DTransform()
        euler21.SetParameters((0, 0, -0.2617993877991494, -60, 0, 0))
        euler21.SetFixedParameters((0, -15.169497244391824, 0,0))
        euler22 = sitk.Euler3DTransform()
        euler22.SetParameters((0, 0, 0.06044069380049563 + params[0, 1], -3.6672184664993104 + params[1, 1], 5.519589055438036 + params[2, 1], 0))
        euler22.SetFixedParameters((19.20823810000698 + params[1, 1], 71.71820919916134 + params[2, 1], 0,0))

        composite2.AddTransform(euler21)
        composite2.AddTransform(euler22)
        self._transforms[1] = composite2

    def _prepare_transforms3(self, images, params, flip):
        """
            Default transform for the alignment of three images.
        """

        # composite1 = sitk.Transform(self._dimension, sitk.sitkComposite)
        composite1 = sitk.CompositeTransform(self._dimension)

        if flip[0]:
            flipping = sitk.AffineTransform(self._dimension)
            center = np.asarray(images[0].GetOrigin()) + np.multiply(np.asarray(images[0].GetSpacing()),
                                                                     np.asarray(images[0].GetSize())) / 2
            if images[0].GetSize()[2] == 1:
                center[2] = 0
            flipping.SetCenter(center)
            matrix = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
            flipping.SetMatrix(matrix.ravel())

            composite1.AddTransform(flipping)

        euler11 = sitk.Euler3DTransform()
        euler11.SetParameters((0, 0, 0.2617993877991494, 50, 0, 0))
        euler11.SetFixedParameters((0.0003848705291886745, 100.99294164776802, 0, 0))
        euler12 = sitk.Euler3DTransform()
        euler12.SetParameters((0, 0, 0.18861035521910807 + params[0, 0], -5.118717561114299 + params[1, 0], 3.1769646696079907 + params[2, 0], 0))
        euler12.SetFixedParameters((-48.31883116758486 + params[1, 0], 113.81368200441585 + params[2, 0], 0,0))

        composite1.AddTransform(euler11)
        composite1.AddTransform(euler12)
        self._transforms[0] = composite1

        # ------------------------------------------------------------------------------------------------------------

        # composite2 = sitk.Transform(self._dimension, sitk.sitkComposite)
        composite2 = sitk.CompositeTransform(self._dimension)

        if flip[1]:
            flipping = sitk.AffineTransform(self._dimension)
            center = np.asarray(images[1].GetOrigin()) + np.multiply(np.asarray(images[1].GetSpacing()),
                                                                     np.asarray(images[1].GetSize())) / 2
            if images[1].GetSize()[2] == 1:
                center[2] = 0
            flipping.SetCenter(center)
            matrix = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
            flipping.SetMatrix(matrix.ravel())

            composite2.AddTransform(flipping)

        euler21 = sitk.Euler3DTransform()
        euler21.SetParameters((0, 0, 0, 0, 0, 0))
        euler21.SetFixedParameters((0.0003848705291886745, 100.99294164776802, 0, 0))
        euler22 = sitk.Euler3DTransform()
        euler22.SetParameters((0, 0, params[0, 1], params[1, 1], params[2, 1], 0))
        euler22.SetFixedParameters((0.0003848705291886745 + params[1, 1], 100.99294164776802 + params[2, 1], 0,0))

        composite2.AddTransform(euler21)
        composite2.AddTransform(euler22)
        self._transforms[1] = composite2

        # ------------------------------------------------------------------------------------------------------------

        # composite3 = sitk.Transform(self._dimension, sitk.sitkComposite)
        composite3 = sitk.CompositeTransform(self._dimension)

        if flip[2]:
            flipping = sitk.AffineTransform(self._dimension)
            center = np.asarray(images[2].GetOrigin()) + np.multiply(np.asarray(images[2].GetSpacing()),
                                                                     np.asarray(images[2].GetSize())) / 2
            if images[2].GetSize()[2] == 1:
                center[2] = 0
            flipping.SetCenter(center)
            matrix = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
            flipping.SetMatrix(matrix.ravel())

            composite3.AddTransform(flipping)

        euler31 = sitk.Euler3DTransform()
        euler31.SetParameters((0, 0, -0.2617993877991494, -50, 0, 0))
        euler31.SetFixedParameters((0.0003848705291886745, 100.99294164776802, 0, 0))
        euler32 = sitk.Euler3DTransform()
        euler32.SetParameters((0, 0, -0.17854658142069846 + params[0, 2], 5.039741610838157 + params[1, 2], 1.7086000852233232 + params[2, 2], 0))
        euler32.SetFixedParameters((48.272217255972684 + params[1, 2], 113.812913678203 + params[2, 2], 0, 0))

        composite3.AddTransform(euler31)
        composite3.AddTransform(euler32)
        self._transforms[2] = composite3

    def _get_new_bounds(self, images):
        """
            Calculate the image bounds of the fused image.
        """

        self._new_spacing = images[0].GetSpacing()
        new_bounds = []
        for i in range(0,len(images)):
            # get image bounds
            corners = self._get_bounding_box(images[i])

            for c in range(len(corners)):
                new_bounds.append(self._transforms[i].GetInverse().TransformPoint(corners[c]))

        min_bounds = np.amin(new_bounds, axis=0)
        max_bounds = np.amax(new_bounds, axis=0)

        self._new_origin = tuple(min_bounds)
        if self._dimension==2:
            self._new_size = (int((max_bounds[0]-min_bounds[0])/self._new_spacing[0]),
                    int((max_bounds[1]-min_bounds[1])/self._new_spacing[1]))
        elif self._dimension==3:
            if images[0].GetSize()[2] > 1:
                self._new_size = (int((max_bounds[0] - min_bounds[0]) / self._new_spacing[0]),
                        int((max_bounds[1] - min_bounds[1]) / self._new_spacing[1]),
                        int((max_bounds[2] - min_bounds[2]) / self._new_spacing[2]))
            else:
                self._new_size = (int((max_bounds[0] - min_bounds[0]) / self._new_spacing[0]),
                                  int((max_bounds[1] - min_bounds[1]) / self._new_spacing[1]), 1)

        # return self._new_origin, self._new_size, self._new_spacing

    def _get_bounding_box(self, image):
        """
            Get the bounding box of an image.
        """

        inside_value = 0
        outside_value = 1 # 255

        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        # label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
        label_shape_filter.Execute(image)
        # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
        bounding_box = label_shape_filter.GetBoundingBox(outside_value)

        corners = []
        if self._dimension==2:
            corners = [image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1])),
                       image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1])),
                       image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1]+bounding_box[4]-1)),
                       image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1]+bounding_box[4]-1))]
        elif self._dimension==3:
            corners = [image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1], bounding_box[2])),
                       image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1], bounding_box[2])),
                       image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1]+bounding_box[4]-1, bounding_box[2])),
                       image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1]+bounding_box[4]-1, bounding_box[2])),
                       image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1], bounding_box[2]+bounding_box[5]-1)),
                       image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1], bounding_box[2]+bounding_box[5]-1)),
                       image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1]+bounding_box[4]-1, bounding_box[2]+bounding_box[5]-1)),
                       image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1]+bounding_box[4]-1, bounding_box[2]+bounding_box[5]-1))]

        return corners


class MaximumImageFusion(ImageFusion):
    """
        Maximum intensity fusion of two or three ultrasound images.
        Inherits from base class ImageFusion.

        Arguments (Base class)
        ---------
        `dimension` : image dimension
        `number_images` : number of images to fuse
        `interp` : string or list/tuple of string
           possible values from this set: {'linear', 'nearest', 'bspline'}
           Different types of interpolation can be provided for each input,
           e.g. for two inputs, `interp=['linear','nearest']
        --------

        Extra options
        -------

    """
    def __init__(self, dimension=3, number_images=2, interp='linear'):
        super(MaximumImageFusion, self).__init__(dimension=dimension, number_images=number_images, interp=interp)

    """
        This function has to be overwritten by classes inheriting from ImageFusion!
    """
    def _fuse(self, images):
        transformed, transformed_masks = super()._fuse(images)

        fused = sitk.GetArrayFromImage(transformed[0])
        for i in range(1, len(transformed)):
            img = sitk.GetArrayFromImage(transformed[i])
            fused = np.maximum(fused, img)

        fused_image = sitk.GetImageFromArray(fused, isVector=False)
        fused_image.SetSpacing(transformed[0].GetSpacing())
        fused_image.SetOrigin(transformed[0].GetOrigin())

        return fused_image, transformed_masks


class AverageImageFusion(ImageFusion):
    """
        Average intensity fusion of two or three ultrasound images.
        Inherits from base class ImageFusion.

        Arguments (Base class)
        ---------
        `dimension` : image dimension
        `number_images` : number of images to fuse
        `interp` : string or list/tuple of string
           possible values from this set: {'linear', 'nearest', 'bspline'}
           Different types of interpolation can be provided for each input,
           e.g. for two inputs, `interp=['linear','nearest']
        --------

        Extra options
        -------

    """
    def __init__(self, dimension=3, number_images=2, interp='linear'):
        super(AverageImageFusion, self).__init__(dimension=dimension, number_images=number_images, interp=interp)

    """
        Set/Get functions.
    """
    """
        This function has to be overwritten by classes inheriting from ImageFusion!
    """
    def _fuse(self, images):
        transformed, transformed_masks = super()._fuse(images)

        # fused = sitk.GetArrayFromImage(transformed[0])
        mask = sitk.GetArrayFromImage(transformed_masks[0])

        fused = sitk.GetArrayFromImage(sitk.Cast(transformed[0], sitk.sitkFloat32))
        mask = sitk.GetArrayFromImage(sitk.Cast(transformed_masks[0], sitk.sitkFloat32))

        for i in range(1, len(transformed)):
            # fused = fused + sitk.GetArrayFromImage(transformed[i])
            # mask = mask + sitk.GetArrayFromImage(transformed_masks[i])
            fused = fused + sitk.GetArrayFromImage(sitk.Cast(transformed[i], sitk.sitkFloat32))
            mask = mask + sitk.GetArrayFromImage(sitk.Cast(transformed_masks[i], sitk.sitkFloat32))
        mask[np.where(mask == 0)] = 1
        fused = np.divide(fused, mask)

        fused_image = sitk.GetImageFromArray(fused, isVector=False)
        fused_image.SetSpacing(transformed[0].GetSpacing())
        fused_image.SetOrigin(transformed[0].GetOrigin())

        fused_image = sitk.RescaleIntensity(fused_image, 0, 255)

        return fused_image, transformed_masks


class AdditionImageFusion(ImageFusion):
    """
        Average intensity fusion of two or three ultrasound images.
        Inherits from base class ImageFusion.

        Arguments (Base class)
        ---------
        `dimension` : image dimension
        `number_images` : number of images to fuse
        `interp` : string or list/tuple of string
           possible values from this set: {'linear', 'nearest', 'bspline'}
           Different types of interpolation can be provided for each input,
           e.g. for two inputs, `interp=['linear','nearest']
        --------

        Extra options
        -------

    """
    def __init__(self, dimension=3, number_images=2, interp='linear'):
        super(AdditionImageFusion, self).__init__(dimension=dimension, number_images=number_images, interp=interp)

    """
        Set/Get functions.
    """
    """
        This function has to be overwritten by classes inheriting from ImageFusion!
    """
    def _fuse(self, images):
        transformed, transformed_masks = super()._fuse(images)

        # fused = sitk.GetArrayFromImage(transformed[0])
        mask = sitk.GetArrayFromImage(transformed_masks[0])

        fused = sitk.GetArrayFromImage(sitk.Cast(transformed[0], sitk.sitkFloat32))
        mask = sitk.GetArrayFromImage(sitk.Cast(transformed_masks[0], sitk.sitkFloat32))

        for i in range(1, len(transformed)):
            # fused = fused + sitk.GetArrayFromImage(transformed[i])
            # mask = mask + sitk.GetArrayFromImage(transformed_masks[i])
            fused = fused + sitk.GetArrayFromImage(sitk.Cast(transformed[i], sitk.sitkFloat32))
            mask = mask + sitk.GetArrayFromImage(sitk.Cast(transformed_masks[i], sitk.sitkFloat32))
        mask[np.where(mask == 0)] = 1
        # fused = np.divide(fused, mask)

        fused_image = sitk.GetImageFromArray(fused, isVector=False)
        fused_image.SetSpacing(transformed[0].GetSpacing())
        fused_image.SetOrigin(transformed[0].GetOrigin())

        fused_image = sitk.RescaleIntensity(fused_image, 0, 255)

        return fused_image, transformed_masks


class WeightedFrustumImageFusion(ImageFusion):
    """
        Weighted intensity fusion of two or three ultrasound images.
        Inherits from base class ImageFusion.

        Arguments (Base class)
        ---------
        `dimension` : image dimension
        `number_images` : number of images to fuse
        `interp` : string or list/tuple of string
          possible values from this set: {'linear', 'nearest', 'bspline'}
          Different types of interpolation can be provided for each input,
          e.g. for two inputs, `interp=['linear','nearest']
        --------

        Extra options
        -------
        `weights` : type of weights {'gauss', 'gauss+signal'}
                    'gauss' : frustum weighting using 3D gaussian.
                    'gauss+signal' : frustum weighting using 3D gaussian and signal intensity (used in my MICCAI_2019 paper).
        `sigma` : (vector of) sigmas for gaussian
        `center` : center of gaussian

    """
    def __init__(self, dimension=3, number_images=2, interp='linear'):
        super(WeightedFrustumImageFusion, self).__init__(dimension=dimension, number_images=number_images, interp=interp)

        self._dimension = dimension
        self._number_images = number_images
        self._interp = interp
        self._sigma = [100, 80, 110]
        self._center = None
        self._weights = 'gauss+signal'
        self._weight_images = None

    """
        Set/Get functions.
    """
    def set_weights(self, arg):
        assert arg == 'gauss' or arg == 'gauss+signal', "wrong weight type '{}', only 'gauss', 'gauss+signal' supported.".format(arg)
        self._weights = arg

    def set_sigma(self, arg):
        assert len(arg) == self._dimension, 'wrong vector length: Expected {:d}, got {:d}.'.format(self._dimension, len(arg))
        self._sigma = arg

    def set_center(self, arg):
        assert len(arg) == self._dimension, 'wrong vector length: Expected {:d}, got {:d}.'.format(self._dimension, len(arg))
        self._center = arg

    def get_weights(self):
        return self._weights

    def get_sigma(self):
        return self._sigma

    def get_center(self):
        return self._center

    def get_weight_images(self):
        assert self._weight_images is not None, 'Weight images have not been created yet.'
        return self._weight_images

    """
        This function has to be overwritten by classes inheriting from ImageFusion!
    """
    def _fuse(self, images):

        transformed, transformed_masks = super()._fuse(images)

        weights = self._create_weight_images(images)

        transformed = self._prepare_images(images)
        self._weight_images = self._prepare_images(weights)

        sum_weights = sitk.GetArrayFromImage(self._weight_images[0])
        for i in range(1, len(self._weight_images)):
            sum_weights = sum_weights + sitk.GetArrayFromImage(self._weight_images[i])

        with np.errstate(divide='ignore', invalid='ignore'):
            fused = np.divide(np.multiply(sitk.GetArrayFromImage(self._weight_images[0]), sitk.GetArrayFromImage(transformed[0])), sum_weights)
            for i in range(1, len(self._weight_images)):
                fused = fused + np.divide(np.multiply(sitk.GetArrayFromImage(self._weight_images[i]), sitk.GetArrayFromImage(transformed[i])), sum_weights)

        fused = np.nan_to_num(fused)
        fused_image = sitk.GetImageFromArray(fused, isVector=False)
        fused_image.SetSpacing(transformed[0].GetSpacing())
        fused_image.SetOrigin(transformed[0].GetOrigin())

        return fused_image, transformed_masks

    """
        Internal functions.
    """
    def _create_weight_images(self, images):

        print("Create weight images")
        weight_images = []
        for i in range(0, len(images)):

            polar_size, polar_spacing, polar_origin, metadata = cart2pol_info(images[i], (0, 0, 0))

            # if i==0:
            #     print('Weight type: {}'.format(self._weights))
            #     print('Sigma: ')
            #     print(self._sigma)

            # gaussian weights
            maxsize = np.max(images[i].GetSize())
            if self._center is not None:
                gauss3d = makeGaussian(3, maxsize, fwhm=self._sigma, center=self._center)
            else:
                gauss3d = makeGaussian(3, maxsize, fwhm=self._sigma)
            gauss3d_image = sitk.GetImageFromArray(gauss3d)
            gauss = rutils.image_resample(gauss3d_image, out_size=polar_size, interp=self._interp)
            gauss.SetOrigin(polar_origin)
            gauss.SetSpacing(polar_spacing)
            gauss = sitk.RescaleIntensity(gauss)
            gauss = self._set_metadata(gauss, images[i])
            gauss = self._set_metadata_polar(gauss, images[i], metadata)

            print('Depth far: ' + str(metadata['depthfar']))

            cartesian_gauss = pol2cart(gauss)
            cartesian_gauss = rutils.image_resample(cartesian_gauss, reference=images[i], interp=self._interp)

            if self._weights == 'gauss+signal':
                weights = np.multiply(sitk.GetArrayFromImage(sitk.Cast(images[i], sitk.sitkFloat32)), sitk.GetArrayFromImage(sitk.Cast(cartesian_gauss, sitk.sitkFloat32)))
            else:
                weights = sitk.GetArrayFromImage(sitk.Cast(cartesian_gauss, sitk.sitkFloat32))

            weights[np.where(weights == 0)] = 0.0001

            weights = np.multiply(weights, sitk.GetArrayFromImage(self._image_masks[i]))

            weight_image = sitk.GetImageFromArray(weights)
            weight_image.SetOrigin(images[i].GetOrigin())
            weight_image.SetSpacing(images[i].GetSpacing())

            weight_images.append(weight_image)

        return weight_images

    def _set_metadata_polar(self, image, reference, metadata):

        image.SetMetaData('r_internal', str(metadata['r_internal']))

        if reference.GetDimension() > 2 and reference.GetSize()[2] > 1:
            # depthnear and depthfar are assumed to be on either side of the beam 'focus'
            image.SetMetaData('SectorStartDepth', str(metadata['depthnear'] / 10.))
            image.SetMetaData('SectorStopDepth', str(metadata['depthfar'] / 10.))
            # sectorwidth is calculated assuming the sides of the frustum (in the x dim) touch the image borders
            # image.SetMetaData('SectorWidth2', str(np.rad2deg(metadata['sectorwidth'])))
            image.SetMetaData('SectorWidth2', str(0))
            # sectorpan   is calculated assuming the sides of the frustum (in the z dim) touch the image borders
            image.SetMetaData('SectorPan', str(np.rad2deg(metadata['sectorpan'])))
            # when in 3D mode the sectorangle metadat is corrupted, impose zero:
            image.SetMetaData('SectorAngle', str(metadata['sectorangle']))
            # spacing in the Cartesian coordinates
            image.SetMetaData('PhysicalDeltaX', str(reference.GetSpacing()[0] / 10.))
            image.SetMetaData('PhysicalDeltaY', str(reference.GetSpacing()[1] / 10.))
            image.SetMetaData('PhysicalDeltaZ', str(reference.GetSpacing()[2] / 10.))

        return image
