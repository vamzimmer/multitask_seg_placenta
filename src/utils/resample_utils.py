import math
import SimpleITK as sitk
import numpy as np


def image_resample(image, downsample=0, transform=None, reference=None, out_origin=None, out_size=None,
                   out_spacing=None, out_direction=None, interp='linear', default_intensity=0):
    """
    Resample an ITK image to either:
    a reference image given by an ITK image or,
    a desired voxel spacing in mm given by [spXmm, spYmm, spZmm] or,
    a desired size [x, y, z], or
    a downsampling factor (applied to the voxel spacing)

    Arguments
    ---------
    `reference` : ITK image
    `downsample` : scalar
    `out_spacing` : tuple or list (e.g [1.,1.,1.])
        New spacing in mm.
    `out_size` : tuple or list of ints (e.g. [100, 100, 100])
    `interp` : string or list/tuple of string
        possible values from this set: {'linear', 'nearest', 'bspline'}
        Different types of interpolation can be provided for each input,
        e.g. for two inputs, `interp=['linear','nearest']

    """
    in_size = image.GetSize()
    in_spacing = image.GetSpacing()
    in_origin = image.GetOrigin()
    in_direction = image.GetDirection()
    # if not downsample and reference is None and out_spacing is None and out_size is None:
    #     out_size = image.GetSize()
    #     out_spacing = image.GetSpacing()

    if not downsample:
        if reference is not None:
            out_spacing = reference.GetSpacing()
            out_size = reference.GetSize()
            out_origin = reference.GetOrigin()
            out_direction = reference.GetDirection()
        elif out_size is not None and out_spacing is None:
            out_spacing = []
            for d in range(image.GetDimension()):
                out_spacing.append(in_spacing[d] * in_size[d] / out_size[d])
        elif out_spacing is not None and out_size is None:
            out_size = []
            for d in range(image.GetDimension()):
                out_size.append(int(math.ceil(in_size[d] * (in_spacing[d] / out_spacing[d]))))
    else:
        out_spacing = []
        for d in range(image.GetDimension()):
            out_spacing.append(in_spacing[d] * downsample)
        out_size = []
        for d in range(image.GetDimension()):
            out_size.append(int(math.ceil(in_size[d] * (in_spacing[d] / out_spacing[d]))))

    if out_origin is None:
        out_origin = in_origin
    if out_spacing is None:
        out_spacing = in_spacing
    if out_size is None:
        out_size = in_size
    if out_direction is None:
        out_direction = in_direction

    if transform is None:
        transform = sitk.Transform()

    if interp == 'linear':
        interp_func = sitk.sitkLinear
    elif interp == 'nearest':
        interp_func = sitk.sitkNearestNeighbor
    elif interp == 'bspline':
        interp_func = sitk.sitkBSpline
    else:
        assert False, "wong interpolation method  (" + interp + "). only linear, nearest and bspline interpolation supported"

    resampled_img = sitk.Resample(image, out_size, transform, interp_func, out_origin, out_spacing,
                                  out_direction, default_intensity, image.GetPixelIDValue())

    return resampled_img


def transform_image(image, transform, reference_image=None, new_bounds=False, image_mask=None, interp='linear', default_intensity=0):
    """
    Transform an itk image using an itk transform.
    If new_bounds, the size and origin of image are adapted to transformation

    Arguments
    ---------
    `interp` : string or list/tuple of string
        possible values from this set: {'linear', 'nearest', 'bspline'}
        Different types of interpolation can be provided for each input,
        e.g. for two inputs, `interp=['linear','nearest']

    """
    in_size = image.GetSize()
    in_spacing = image.GetSpacing()
    in_origin = image.GetOrigin()
    # if not downsample and reference is None and out_spacing is None and out_size is None:
    #     out_size = image.GetSize()
    #     out_spacing = image.GetSpacing()

    if reference_image is None:
        if new_bounds:
            if image_mask is not None:
                out_origin, out_size, out_spacing = get_new_bounds([image_mask], [transform])
            else:
                out_origin, out_size, out_spacing = get_new_bounds([image], [transform])
        else:
            out_origin = in_origin
            out_spacing = in_spacing
            out_size = in_size
    else:
        out_spacing = reference_image.GetSpacing()
        out_size = reference_image.GetSize()
        out_origin = reference_image.GetOrigin()

    if interp == 'linear':
        interp_func = sitk.sitkLinear
    elif interp == 'nearest':
        interp_func = sitk.sitkNearestNeighbor
    elif interp == 'bspline':
        interp_func = sitk.sitkBSpline
    else:
        assert False, "wong interpolation method  (" + interp + "). only linear, nearest and bspline interpolation supported"
    # print(image.GetSize())
    # print(out_size)
    # resampled_img = sitk.Resample(image, ref, transform, interp_func,
    #                               image.GetDirection(), default_intensity, image.GetPixelIDValue())
    resampled_img = sitk.Resample(image, out_size, transform, interp_func, out_origin, out_spacing,
                                  image.GetDirection(), default_intensity, image.GetPixelIDValue())
    return resampled_img


def get_bounding_box(image, dimension=3):
    """
        Get the bounding box of an image.
    """

    inside_value = 0
    outside_value = 255

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)

    corners = []
    if dimension == 2:
        corners = [image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1])),
                   image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1])),
                   image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1]+bounding_box[4]-1)),
                   image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1]+bounding_box[4]-1))]
    elif dimension == 3:
        corners = [image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1], bounding_box[2])),
                   image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1], bounding_box[2])),
                   image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1]+bounding_box[4]-1, bounding_box[2])),
                   image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1]+bounding_box[4]-1, bounding_box[2])),
                   image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1], bounding_box[2]+bounding_box[5]-1)),
                   image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1], bounding_box[2]+bounding_box[5]-1)),
                   image.TransformIndexToPhysicalPoint((bounding_box[0], bounding_box[1]+bounding_box[4]-1, bounding_box[2]+bounding_box[5]-1)),
                   image.TransformIndexToPhysicalPoint((bounding_box[0]+bounding_box[3]-1, bounding_box[1]+bounding_box[4]-1, bounding_box[2]+bounding_box[5]-1))]

    return corners


def get_new_bounds(images, transforms, dimension=3):
    """
        Calculate the image bounds of the fused image.
    """

    new_spacing = images[0].GetSpacing()
    new_bounds = []
    for i in range(0,len(images)):
        # get image bounds
        # mask = iutils.create_frustum_mask(images[i])
        corners = get_bounding_box(images[i])

        for c in range(len(corners)):
            new_bounds.append(transforms[i].GetInverse().TransformPoint(corners[c]))

    min_bounds = np.amin(new_bounds, axis=0)
    max_bounds = np.amax(new_bounds, axis=0)

    new_origin = tuple(min_bounds)
    if dimension==2:
        new_size = (int((max_bounds[0]-min_bounds[0])/new_spacing[0]),
                int((max_bounds[1]-min_bounds[1])/new_spacing[1]))
    elif dimension==3:
        if images[0].GetSize()[2] > 1:
            new_size = (int((max_bounds[0] - min_bounds[0]) / new_spacing[0]),
                    int((max_bounds[1] - min_bounds[1]) / new_spacing[1]),
                    int((max_bounds[2] - min_bounds[2]) / new_spacing[2]))
        else:
            new_size = (int((max_bounds[0] - min_bounds[0]) / new_spacing[0]),
                              int((max_bounds[1] - min_bounds[1]) / new_spacing[1]), 1)

    return new_origin, new_size, new_spacing
