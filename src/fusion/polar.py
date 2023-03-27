import sys
import os
import numpy as np
import math
from matplotlib import pyplot as plt
import SimpleITK as sitk

"""
@Nicolas Toussaint
@Veronika A Zimmer
"""

def get_cartesian_representation(image, spacing, reference):

    all_keys = reference.GetMetaDataKeys()
    for key in all_keys:
        image.SetMetaData(key, reference.GetMetaData(key))
    image.SetSpacing(spacing)

    in_im_2 = pol2cart(image)

    return in_im_2


def calculate_frustum(in_image):
    """Evaluate the frustum geometry information from the input image metadata

    Keyword arguments:
    in_image  -- input image (simple::Image)

    Returns:
    metadata  -- dictionary containing frustum information


    The input image is assumed to contain the following information in in_image.GetMetaData() dictionary:

    - SectorStartDepth - distance from transducer tip to first 'row' (given in cm, converted to mm)
    - SectorStopDepth  - distance from transducer tip to last 'row' (given in cm, converted to mm)
    - SectorWidth2     - frustum total angle in the x direction (given in deg, converted to rad) ## only in 2D mode
    - FocusDepth       - distance from transducer tip to the beam 'focus' (given in cm, converted to mm)
    - SectorAngle      - tip angle (in radian)
    - Zoom             - zoom function (1 if activated, 0 of not)
    - TransducerData   - transducer type (X6_2, C9_2, C5_9, etc)

    the output metadata contains the input information, plus additional keywords:

    - sectorpan      - sectorpan is calculated assuming the sides of the frustum (in the z dim) touch the image borders
    - r_internal     - distance between frustum center and transducer tip (in mm) ## ad-hoc
    - center         - coordinates of the center of the frustum with respect to the input image
    - theta1, theta2 - start and end angles in the x direction (in rad)
    - phi1, phi2     - start and end angles in the x direction (in rad)
    - r1, r2         - start and end radii of the frustum (in mm)

    In 3D mode, SectorStartDepth, SectorStopDepth, SectorWidth2 and SectorAngle are not provided.
    Therefore they are derived from other parameters, namely 'FocusDepth' and the image dimension and spacing:

    - 'depthnear'   = 'focusdepth' - image_size[1] / 2
    - 'depthfar'    = 'focusdepth' + image_size[1] / 2
    - 'sectorwidth' = 2 arcsin( image_size[0] / ('r_internal' + 'depthfar') )
    - 'sectorpan'   = 2 arcsin( image_size[2] / ('r_internal' + 'depthfar') )
    - 'sectorangle' = 0.


    """
    # Query input image geometry
    in_size = in_image.GetSize()
    in_spacing = in_image.GetSpacing()
    in_dimension = in_image.GetDimension()
    halfimagesize = [0.5 * in_size[idx] * in_spacing[idx] for idx in range(in_dimension)]

    # query metadata information for transformation
    metadata = {}
    ## distance from transducer tip to first 'row' (given in cm, converted to mm)
    metadata['depthnear'] = 10. * float(
        in_image.GetMetaData('SectorStartDepth')) if 'SectorStartDepth' in in_image.GetMetaDataKeys() else -1
    ## distance from transducer tip to last 'row' (given in cm, converted to mm)
    metadata['depthfar'] = 10. * float(
        in_image.GetMetaData('SectorStopDepth')) if 'SectorStopDepth' in in_image.GetMetaDataKeys() else -1
    ## frustum total angle in the x direction (given in deg, converted to rad) ## only in 2D mode
    metadata['sectorwidth'] = np.deg2rad(
        float(in_image.GetMetaData('SectorWidth2'))) if 'SectorWidth2' in in_image.GetMetaDataKeys() else -1
    ## frustum total angle in the z direction (given in deg, converted to rad) ## only in 3D mode
    metadata['sectorpan'] = np.deg2rad(
        float(in_image.GetMetaData('SectorPan'))) if 'SectorPan' in in_image.GetMetaDataKeys() else 0.
    ## distance from transducer tip to the beam 'focus' (given in cm, converted to mm)
    metadata['focusdepth'] = 10. * float(
        in_image.GetMetaData('FocusDepth')) if 'FocusDepth' in in_image.GetMetaDataKeys() else halfimagesize[1] / 10.
    ## tip angle (in radian)
    metadata['sectorangle'] = float(
        in_image.GetMetaData('SectorAngle')) if 'SectorAngle' in in_image.GetMetaDataKeys() else 0.
    ## zoom function (1 if activated, 0 of not)
    metadata['zoom'] = float(in_image.GetMetaData('Zoom')) if 'Zoom' in in_image.GetMetaDataKeys() else -1
    ## transducer type (X6_2, C9_2, C5_9, etc)
    metadata['transducerdata'] = in_image.GetMetaData(
        'TransducerData') if 'TransducerData' in in_image.GetMetaDataKeys() else 'N/A'
    ## distance between frustum center and transducer tip (in mm) ## ad-hoc
    metadata['r_internal'] = -1.
    if 'X' in metadata['transducerdata']:
        metadata['r_internal'] = 20.
    elif 'C' in metadata['transducerdata']:
        metadata['r_internal'] = 38.
    elif 'O' in metadata['transducerdata']:
        metadata['r_internal'] = 57.

    ## coordinates of the center of the frustum with respect to the input image
    metadata['center'] = [0.,
                          - (metadata['r_internal'] + metadata['depthnear']) * np.cos(metadata['sectorwidth'] / 2.0)]

    if in_dimension > 2 and in_size[2] > 1:
        ## depthnear and depthfar are assumed to be on either side of the beam 'focus'
        metadata['depthnear'] = metadata['focusdepth'] - halfimagesize[1]
        metadata['depthfar'] = metadata['focusdepth'] + halfimagesize[1]
        ## sectorwidth is calculated assuming the sides of the frustum (in the x dim) touch the image borders
        metadata['sectorwidth'] = 2.0 * np.arcsin(halfimagesize[0] / (metadata['r_internal'] + metadata['depthfar']))
        ## sectorpan   is calculated assuming the sides of the frustum (in the z dim) touch the image borders
        metadata['sectorpan'] = 2.0 * np.arcsin(halfimagesize[2] / (metadata['r_internal'] + metadata['depthfar']))
        ## coordinates of the center of the frustum with respect to the input image
        metadata['center'] = [0., - (metadata['r_internal'] + metadata['depthnear']) * np.cos(
            metadata['sectorwidth'] / 2.0), 0]
        ## when in 3D mode the sectorangle metadat is corrupted, impose zero:
        metadata['sectorangle'] = 0.

    ## theta1 and theta2 are the start and end angles in the x direction (in rad)
    metadata['theta1'] = metadata['sectorangle'] - metadata['sectorwidth'] / 2.0
    metadata['theta2'] = metadata['sectorangle'] + metadata['sectorwidth'] / 2.0

    ## theta1 and theta2 are the start and end angles in the x direction (in rad)
    metadata['phi1'] = 0. - metadata['sectorpan'] / 2.0
    metadata['phi2'] = 0. + metadata['sectorpan'] / 2.0

    ## r1 and r2 are the start and end radii of the frustum (in mm)
    metadata['r1'] = metadata['r_internal'] + metadata['depthnear']
    metadata['r2'] = metadata['r_internal'] + metadata['depthfar']



    return metadata


def calculate_cartesian_box(in_image):
    in_size = in_image.GetSize()
    in_spacing = in_image.GetSpacing()
    in_dimension = in_image.GetDimension()
    in_origin = in_image.GetOrigin()
    halfimagesize = [0.5 * in_size[idx] * in_spacing[idx] for idx in range(in_dimension)]

    # query metadata information for transformation
    metadata = {}
    ## number of pixels in the x direction
    metadata['cols'] = int(in_image.GetMetaData('Columns')) if 'Columns' in in_image.GetMetaDataKeys() else 0
    ## number of pixels in the y direction
    metadata['rows'] = int(in_image.GetMetaData('Rows')) if 'Rows' in in_image.GetMetaDataKeys() else 0
    ## number of pixels in the z direction ## only in 3D
    depth_0 = int(
        in_image.GetMetaData('RegionLocationMinz0')) if 'RegionLocationMinz0' in in_image.GetMetaDataKeys() else 0
    depth_1 = int(
        in_image.GetMetaData('RegionLocationMaxz1')) if 'RegionLocationMaxz1' in in_image.GetMetaDataKeys() else 1
    metadata['depth'] = depth_1 - depth_0
    ## distance from transducer tip to first 'row' (given in cm, converted to mm)
    metadata['depthnear'] = 10. * float(
        in_image.GetMetaData('SectorStartDepth')) if 'SectorStartDepth' in in_image.GetMetaDataKeys() else -1
    ## distance from transducer tip to last 'row' (given in cm, converted to mm)
    metadata['depthfar'] = 10. * float(
        in_image.GetMetaData('SectorStopDepth')) if 'SectorStopDepth' in in_image.GetMetaDataKeys() else -1
    ## frustum total angle in the x direction (given in deg, converted to rad) ## only in 2D mode
    metadata['sectorwidth'] = np.deg2rad(
        float(in_image.GetMetaData('SectorWidth2'))) if 'SectorWidth2' in in_image.GetMetaDataKeys() else -1

    ## spacing in x direction ## only in 2D, fed from calculate_frustum otherwise
    metadata['dx'] = 10. * float(
        in_image.GetMetaData('PhysicalDeltaX')) if 'PhysicalDeltaX' in in_image.GetMetaDataKeys() else 1
    ## spacing in y direction ## only in 2D, fed from calculate_frustum otherwise
    metadata['dy'] = 10. * float(
        in_image.GetMetaData('PhysicalDeltaY')) if 'PhysicalDeltaY' in in_image.GetMetaDataKeys() else 1
    ## spacing in z direction ## only in 2D, fed from calculate_frustum otherwise
    metadata['dz'] = 10. * float(
        in_image.GetMetaData('PhysicalDeltaZ')) if 'PhysicalDeltaZ' in in_image.GetMetaDataKeys() else 1

    ## distance between frustum center and transducer tip (in mm) ## ad-hoc
    metadata['r_internal'] = float(
        in_image.GetMetaData('r_internal')) if 'r_internal' in in_image.GetMetaDataKeys() else 0

    ## coordinates of the center of the frustum with respect to the input image
    metadata['center'] = [0.,
                          - (metadata['r_internal'] + metadata['depthnear']) * np.cos(metadata['sectorwidth'] / 2.0)]

    imagesize = [0] * 3
    spacing = [0] * 3
    origin = [0] * 3

    if in_image.GetSize()[2] <= 1:

        # image size
        imagesize[0] = metadata['cols']
        imagesize[1] = metadata['rows']
        imagesize[2] = 1

        # image spacing
        spacing[0] = metadata['dx']
        spacing[1] = metadata['dy']
        spacing[2] = 1

        # image origin
        origin[0] = - (imagesize[0] - 1) * spacing[0] / 2.
        origin[1] = - metadata['center'][1]
        origin[2] = 0.

    else:

        # image size
        imagesize[0] = metadata['cols']
        imagesize[1] = metadata['rows']
        imagesize[2] = metadata['depth']

        # image spacing
        spacing[0] = metadata['dx']
        spacing[1] = metadata['dy']
        spacing[2] = metadata['dz']

        # image origin
        origin[0] = - (imagesize[0] - 1) * spacing[0] / 2.
        origin[1] = - metadata['center'][1]
        origin[2] = - (imagesize[2] - 1) * spacing[2] / 2.

    metadata['imagesize'] = imagesize
    metadata['spacing'] = spacing
    metadata['origin'] = origin

    return metadata


def pol2cart(in_image, padding=None):
    # extract image geometry
    metadata = calculate_cartesian_box(in_image)

    out_size = metadata['imagesize']
    out_origin = metadata['origin']
    out_spacing = metadata['spacing']

    out_image = sitk.Image(out_size, sitk.sitkInt8)
    out_image.SetOrigin(out_origin)
    out_image.SetSpacing(out_spacing)

    # out_image = resample_image(in_image, out_image, inverse=True)
    out_image = resample_image_fast(in_image, out_image, inverse=True)

    for k in in_image.GetMetaDataKeys():
        out_image.SetMetaData(k, in_image.GetMetaData(k))

    out_origin = list(out_image.GetOrigin())
    out_origin[1] = 0.
    out_image.SetOrigin(out_origin)

    return out_image


def resample_image_fast(in_image, out_image, inverse=False):
    """Resample the input image into the output image space.

    Keyword arguments:
    - in_image  -- input image in Cartesian coordinates
    - out_image -- output image in Spherical coordinates

    Returns:
    - The returned object is a SimpleITK image in spherical coordinates containing pixel information
      from the input image.

    The output image out_image is assumed to be prepared (size, spacing, origin, direction)
    in spherical coordinates.

    The method uses a scipy interpolation approach to speed execution.
    """

    import scipy
    from scipy.interpolate import interpn
    import numpy as np

    def get_image_pixel_positions(itk_img):
        """
        Get the positions of pixels.
        :param itk_img:
        :return:
        """
        spacing = itk_img.GetSpacing()
        origin = itk_img.GetOrigin()
        size = itk_img.GetSize()

        if itk_img.GetDimension() == 3:
            positions = np.mgrid[0.0:size[0], 0.0:size[1], 0.0:size[2]]
            for i in range(itk_img.GetDimension()):
                positions[i, :, :, :] = positions[i, :, :, :] * spacing[i] + origin[i]
        else:
            positions = np.mgrid[0.0:size[0], 0.0:size[1]]
            for i in range(itk_img.GetDimension()):
                positions[i, :, :] = positions[i, :, :] * spacing[i] + origin[i]

        return positions

    # extract image geometry
    metadata = calculate_frustum(in_image)

    # query input image geometry
    in_size = in_image.GetSize()
    in_spacing = in_image.GetSpacing()
    in_dimension = in_image.GetDimension()
    in_origin = in_image.GetOrigin()

    in_points = [np.linspace(0, in_size[0], in_size[0]) * in_spacing[0] + in_origin[0],
                 np.linspace(0, in_size[1], in_size[1]) * in_spacing[1] + in_origin[1],
                 np.linspace(0, in_size[2], in_size[2]) * in_spacing[2] + in_origin[2]]

    transpose_order = (1, 0) if in_image.GetDimension() <= 2 else (2, 1, 0)
    in_values = np.transpose(sitk.GetArrayFromImage(in_image), transpose_order)
    out_points = get_image_pixel_positions(out_image)
    # out_positions = np.vstack(map(np.ravel, out_points))
    out_positions = np.vstack(tuple(map(np.ravel, out_points)))

    transformed_out_positions = transform_points(out_positions, inverse)

    if in_dimension > 2 and in_size[2] == 1:
        in_points = list(in_points)
        in_points.pop(2)
        in_values = in_values[:, :, 0]
        transformed_out_positions = transformed_out_positions[0:2, :]

    interpolated = interpn(in_points, in_values, transformed_out_positions.transpose(),
                           method='nearest', bounds_error=False, fill_value=0.)

    reshaped = interpolated.reshape(out_image.GetSize())
    reshaped = np.array(reshaped, dtype=np.uint8)

    ret_image = sitk.GetImageFromArray(np.transpose(reshaped, transpose_order))
    ret_image.SetOrigin(out_image.GetOrigin())
    ret_image.SetSpacing(out_image.GetSpacing())

    return ret_image


def transform_points(out_pts, inverse=False):
    """Spherical coordinate transformation

    Keyword arguments:
    out_pt  -- input point (iterable) in spherical coordinates

    Returns:
    in_pt   -- output point (iterable) in cartesian coordinates

    x = r sin(theta) cos(phi)
    y = r cos(theta) cos(phi)
    z = r sin(phi)

    r     = sqrt(x^2 + y^2 + z^2)
    theta = tan^-1(x/y)
    phi   = sin^-1(z/r)

    """
    in_pts = np.zeros(out_pts.shape)
    if inverse:
        if (len(out_pts) > 2):
            in_pts[0] = np.linalg.norm(out_pts, axis=0)
            in_pts[1] = np.arctan(out_pts[0] / out_pts[1])
            in_pts[2] = np.arcsin(out_pts[2] / in_pts[0])
        else:
            in_pts[0] = np.linalg.norm(out_pts)
            in_pts[1] = np.arctan(out_pts[0] / out_pts[1])
    else:
        if (len(out_pts) > 2):
            in_pts[0] = out_pts[0] * np.sin(out_pts[1]) * np.cos(out_pts[2])
            in_pts[1] = out_pts[0] * np.cos(out_pts[1]) * np.cos(out_pts[2])
            in_pts[2] = out_pts[0] * np.sin(out_pts[2])
        else:
            in_pts[0] = out_pts[0] * np.sin(out_pts[1])
            in_pts[1] = out_pts[0] * np.cos(out_pts[1])

    return np.array(in_pts)


def extract_slice(in_image, slice_id):
    in_size     = in_image.GetSize()
    out_size    = [in_image.GetSize()[0], in_image.GetSize()[1], 1]
    out_spacing = [in_image.GetSpacing()[0], in_image.GetSpacing()[1], 1]
    out_origin  = [in_image.GetOrigin()[0], in_image.GetOrigin()[1], 0]
    out_image   = sitk.Image(out_size, sitk.sitkUInt8)
    out_image.SetSpacing(out_spacing)
    out_image.SetOrigin(out_origin)
    out_image = sitk.Paste(out_image, in_image, out_size, [0,0,slice_id], [0,0,0])

    for k in in_image.GetMetaDataKeys():
        out_image.SetMetaData(k, in_image.GetMetaData(k))

    return out_image


def cart2pol_info(in_image, padding=None):
    # extract image geometry
    metadata = calculate_frustum(in_image)

    ## query input image geometry
    in_size = in_image.GetSize()
    in_spacing = in_image.GetSpacing()
    in_dimension = in_image.GetDimension()

    if padding is None:
        padding = [0.] * in_image.GetDimension()

    # define output image size, origin, spacing and d irection

    # in_origin = [0.] * in_image.GetDimension()
    # in_origin[0] = - metadata['center'][0] - in_spacing[0] * (in_size[0] - 1.0) / 2.0
    # in_origin[1] = - metadata['center'][1]
    # if in_dimension > 2 and in_size[2] > 1:
    #     in_origin[2] = - metadata['center'][2] - in_spacing[2] * (in_size[2] - 1.0) / 2.0

    out_spacing = [1.] * in_image.GetDimension()
    out_spacing[0] = in_spacing[1]
    out_spacing[1] = (metadata['theta2'] - metadata['theta1']) / (in_size[0])
    if in_dimension > 2 and in_size[2] > 1:
        out_spacing[2] = (metadata['phi2'] - metadata['phi1']) / (in_size[2])

    out_size = [1] * in_image.GetDimension()
    out_size[0] = int((metadata['r2'] - metadata['r1'] + 2. * padding[0]) / out_spacing[0])
    out_size[1] = int((metadata['theta2'] - metadata['theta1'] + 2. * padding[1]) / out_spacing[1])
    if in_dimension > 2 and in_size[2] > 1:
        out_size[2] = int((metadata['phi2'] - metadata['phi1'] + 2. * padding[2]) / out_spacing[2])

    out_origin = [0] * in_image.GetDimension()
    out_origin[0] = metadata['r1'] - padding[0]
    out_origin[1] = metadata['theta1'] - padding[1]
    if in_dimension > 2 and in_size[2] > 1:
        out_origin[2] = metadata['phi1'] - padding[2]

    return out_size, out_spacing, out_origin, metadata