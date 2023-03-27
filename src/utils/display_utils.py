import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# from ..utils import resampling


# colormap for visualizations
mycmap = plt.cm.jet
mycmap._init()
mycmap._lut[:, -1] = np.linspace(0, 0.8, 255 + 4)

mycmap2 = plt.cm.winter
mycmap2._init()
mycmap2._lut[:, -1] = np.linspace(0, 0.8, 255 + 4)

mycmap3 = plt.cm.summer
mycmap3._init()
mycmap3._lut[:, -1] = np.linspace(0, 0.8, 255 + 4)

mycmaps = [mycmap, mycmap2, mycmap3]


def display_2d(images, segs=[], overlay=True,
               transpose=True, slice_number=-1,
               intensity_window=(0, 255),
               title=None, show=True, pfile=None,
               subplots=None, figsize=None):

    if not isinstance(images, list):  # single image
        image = check_itk_image(images, transpose=transpose)

        if isinstance(segs, list) and len(segs) == 0:
            # print("Single image")
            display_image(image, window_min=intensity_window[0], window_max=intensity_window[1],
                          slice_number=slice_number, title=title, show=show, pfile=pfile)
        else:
            if not isinstance(segs, list):
                seg = check_itk_image(segs)
            else:
                seg = check_itk_image(segs[0])
            assert check_image_size(image, seg), 'image and segmentation should have the same dimensions.'

            if not overlay:
                # print("Single image with segmentation contour")
                display_with_contour(image, seg, window_min=intensity_window[0], window_max=intensity_window[1],
                                     title=title, show=show, pfile=pfile)
            else:
                # print("Single image with overlay")
                display_with_overlay(image, seg, title=title, show=show, pfile=pfile, figsize=figsize)
    else:

        if not isinstance(segs, list) or (isinstance(segs, list) and len(segs) == 0):
            # print("Multiple images")
            display_multiple_image(images, window_min=intensity_window[0], window_max=intensity_window[1],
                                   slice_number=slice_number, title=title, show=show, pfile=pfile,
                                   subplots=subplots, figsize=figsize)
        else:
            assert len(images) == len(segs), 'image and segmentation/overlay list should have the same number of element.'

            if not overlay:
                # print("Multiple images with segmentation contour")
                display_multiple_images_with_contour(images, segs, window_min=intensity_window[0], window_max=intensity_window[1],
                                                     slice_number=slice_number, title=title, show=show, pfile=pfile,
                                                     subplots=subplots, figsize=figsize)
            else:
                # print("Multiple images with overlay")
                display_multiple_images_with_overlay(images, segs, slice_number=slice_number, title=title, show=show,
                                                     pfile=pfile, subplots=subplots, figsize=figsize)


def display_3d(images, segs=[], overlay=True,
            transpose=True,
            intensity_window=(None, None),
            title=None, show=True, pfile=None, figsize=None):

    if not isinstance(images, list):  # single image
        image = check_itk_image(images, transpose=transpose)

        if isinstance(segs, list) and len(segs) == 0:
            # print("Single image in 3D")
            display_image_3d(image, title=title, show=show, pfile=pfile)
        else:
            if not isinstance(segs, list):
                seg = check_itk_image(segs)
            else:
                seg = check_itk_image(segs[0])
            assert check_image_size(image, seg), 'image and segmentation should have the same dimensions.'

            if not overlay:
                # print("Single image in 3D with segmentation countour")
                display_with_contour_3d(image, seg, window_min=intensity_window[0], window_max=intensity_window[1],
                                        title=title, show=show, pfile=pfile)
            else:
                # print("Single image in 3D with overlay")
                display_with_overlay_3d(image, seg, title=title, show=show, pfile=pfile)
    else:

        if not isinstance(segs, list) or (isinstance(segs, list) and len(segs) == 0):
            # print("Multiple images")
            display_multiple_image_3d(images, title=title, show=show, pfile=pfile)
        else:
            assert len(images) == len(
                segs), 'image and segmentation/overlay list should have the same number of element.'

            if not overlay:
                # print("Multiple images with segmentation countour")
                display_multiple_image_with_contour_3d(images, segs, window_min=intensity_window[0], window_max=intensity_window[1],
                                                       title=title, show=show, pfile=pfile, figsize=figsize)
            else:
                # print("Multiple images with overlay")
                display_multiple_image_with_overlay_3d(images, segs, title=title, show=show, pfile=pfile, figsize=figsize)


def check_itk_image(image, transpose=True):

    if str(image.__class__) == "<class 'SimpleITK.SimpleITK.Image'>":
        return image
    elif str(image.__class__) == "<class 'numpy.ndarray'>":
        if len(image.shape) > 3:
            image = image.squeeze()
        image = sitk.GetImageFromArray(image)
        return image
    elif str(image.__class__) == "<class 'torch.Tensor'>":
        if len(image.size()) > 3:
            image = image.squeeze()
        if transpose and len(image.size())>2:
            image = sitk.GetImageFromArray(image.permute(2, 1, 0).cpu())
        else:
            image = sitk.GetImageFromArray(image.cpu())
        return image
    # elif str(image.__class__) == "<class 'airlab.utils.image.Image'>":  # image is airlab
    #     image = iutils.image_to_itk(image)
    #     return image
    else:
        print("Image type not supported for display.")


def check_image_size(image1, image2):

    assert len(image1.GetSize()) == len(image2.GetSize()), 'images have different sizes.'

    same_size = True
    for d in range(0, len(image1.GetSize())):
        if not (image1.GetSize()[d] == image2.GetSize()[d]):
            same_size = False
    return same_size


def check_image_spacing(image1, image2):

    assert len(image1.GetSpacing()) == len(image2.GetSpacing()), 'images have different spacings.'

    same_spacing = True
    for d in range(0, len(image1.GetSpacing())):
        if not (image1.GetSpacing()[d] == image2.GetSpacing()[d]):
            same_spacing = False
    return same_spacing


def check_image_origin(image1, image2):

    assert len(image1.GetOrigin()) == len(image2.GetOrigin()), 'images have different origins.'

    same_origin = True
    for d in range(0, len(image1.GetOrigin())):
        if not (image1.GetOrigin()[d] == image2.GetOrigin()[d]):
            same_origin = False
    return same_origin

def check_image_direction(image1, image2):

    assert len(image1.GetDirection()) == len(image2.GetDirection()), 'images have different directions.'

    same_direction = True
    for d in range(0, len(image1.GetDirection())):
        if not (image1.GetDirection()[d] == image2.GetDirection()[d]):
            same_direction = False
    return same_direction


def check_image_space(image1, image2):

    same_size = check_image_size(image1, image2)
    same_spacing = check_image_spacing(image1, image2)
    same_origin = check_image_origin(image1, image2)
    #same_direction = check_image_direction(image1, image2)

    return same_size and same_spacing and same_origin #and same_direction


def display_image(image, slice_number=-1, window_min=0, window_max=255, title=None, show=True, pfile=None):
    """
    Display an image slice.
    """

    image = check_itk_image(image)

    if window_min is None:
        window_min = int(np.amin(image))
    if window_max is None:
        window_max = int(np.amax(image))

    # check image dimension and slice_number
    if len(image.GetSize()) == 2 or image.GetSize()[2] == 1:
        # image is 2D
        img = image
    else:
        if slice_number < 0 or slice_number > image.GetSize()[2]-1:
            slice_number = int(image.GetSize()[2]/2)
        img = image[:, :, slice_number]

    img_np = sitk.GetArrayViewFromImage(img)
    if img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)
    # We assume the original slice is isotropic, otherwise the display would be distorted
    plt.imshow(img_np, cmap='gray', vmin=window_min, vmax=window_max)
    # plt.axis('off')

    if title is not None:
        plt.title(title)
    if pfile is not None:
        plt.savefig(pfile)
    if show:
        plt.show()


def display_multiple_image(images, slice_number=-1, subplots=None, window_min=None, window_max=None, title=None,
                           show=True, pfile=None, figsize=None):
    """
    Display an image slice.
    """
    N = len(images)
    assert title is None or len(title) == N, \
        'Wrong title. Either title=None or one title per image (which can be None).'
    assert subplots is None or (len(subplots) == 2 and subplots[0]*subplots[1]>=N), "Wrong definition of subplots."

    if figsize is None:
        figsize = (15, 6)

    if subplots is None:
        f, a = plt.subplots(1, N, figsize=figsize)
    else:
        f, a = plt.subplots(subplots[0], subplots[1], figsize=figsize)

    for n in range(N):
        if images[n] is not None:
            image = check_itk_image(images[n])

            if window_min is None:
                window_min = int(np.amin(image))
            if window_max is None:
                window_max = int(np.amax(image))

            # check image dimension and slice_number
            if len(image.GetSize()) == 2 or image.GetSize()[2] == 1:
                # image is 2D
                img = image
            else:
                if slice_number < 0 or slice_number > image.GetSize()[2]-1:
                    slice_number = int(image.GetSize()[2]/2)
                img = image[:, :, slice_number]

            img_np = sitk.GetArrayViewFromImage(img)
            if img_np.shape[0] == 1:
                img_np = img_np.squeeze(0)

            if subplots is None:
                # We assume the original slice is isotropic, otherwise the display would be distorted
                a[n].imshow(img_np, cmap='gray', vmin=window_min, vmax=window_max)
                # a[n].axis('off')
            else:
                sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
                a[sub_ind[0], sub_ind[1]].imshow(img_np, cmap='gray', vmin=window_min, vmax=window_max)
                # a[sub_ind[0], sub_ind[1]].axis('off')

            if title is not None:
                if title[n] is not None:
                    if subplots is None:
                        a[n].set_title(title[n])
                    else:
                        a[sub_ind[0], sub_ind[1]].set_title(title[n])
            plt.draw()
        else:
            if subplots is not None:
                # a[n].axis('off')
            # else:
                sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
                # a[sub_ind[0], sub_ind[1]].axis('off')

    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)


def display_with_contour(image, segs, window_min=None, window_max=None, slice_number=-1, title=None, pfile=None, show=True):
    """
    Display an image slice with segmented contours overlaid onto it. The contours are the edges of
    the labeled regions.
    """

    image = check_itk_image(image)
    segs = check_itk_image(segs)

    if window_min is None:
        window_min = int(np.amin(image))
    if window_max is None:
        window_max = int(np.amax(image))

    # check image dimension and slice_number
    if len(image.GetSize()) == 2:
        # image is 2D
        img = image
        msk = segs
    elif image.GetSize()[2] == 1:
        img = image[:, :, 0]
        msk = segs[:, :, 0]
    else:
        if slice_number < 0 or slice_number > image.GetSize()[2] - 1:
            slice_number = int(image.GetSize()[2] / 2)
        img = image[:, :, slice_number]
        msk = segs[:, :, slice_number]

    # if not check_image_space(img, msk):
    #     print("Image and Label do not occupy the same space. Resampling of label...")
    #     msk = resampling.image_resample(msk, reference=img, interp='nearest')

    overlay_img = sitk.LabelMapContourOverlay(sitk.Cast(msk, sitk.sitkLabelUInt8),
                                              sitk.Cast(sitk.IntensityWindowing(img, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8),
                                              opacity=1, contourThickness=[2, 2])

    # We assume the original slice is isotropic, otherwise the display would be distorted
    fig = plt.figure()
    plt.imshow(sitk.GetArrayViewFromImage(overlay_img))
    # plt.axis('off')
    if title is not None:
        plt.title(title)
    if pfile is not None:
        fig.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(fig)


def display_with_overlay(image, maps, slice_number=-1, title=None, pfile=None, show=True, figsize=(15, 6)):
    """
    Display an image slice with segmented contours overlaid onto it. The contours are the edges of
    the labeled regions.
    """

    image = check_itk_image(image)
    map = check_itk_image(maps)

    # check image dimension and slice_number
    if len(image.GetSize()) == 2:
        # image is 2D
        img = image
        mp = map
    elif image.GetSize()[2] == 1:
        img = image[:, :, 0]
        mp = map[:, :, 0]
    else:
        if slice_number < 0 or slice_number > image.GetSize()[2] - 1:
            slice_number = int(image.GetSize()[2] / 2)
        img = image[:, :, slice_number]
        mp = map[:, :, slice_number]

    # We assume the original slice is isotropic, otherwise the display would be distorted
    # fig = plt.figure()
    # plt.imshow(sitk.GetArrayViewFromImage(img), cmap='gray')
    # plt.imshow(sitk.GetArrayViewFromImage(mp), cmap=mycmap, alpha=0.7)

    fig, a = plt.subplots(1, 1, figsize=figsize)
    a.imshow(sitk.GetArrayViewFromImage(img), cmap='gray')
    divider = make_axes_locatable(a)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cim = a.imshow(sitk.GetArrayViewFromImage(mp), cmap=mycmap, alpha=0.7)
    cb0 = fig.colorbar(cim, ax=a, orientation='vertical', cax=cax)
    # plt.draw()

    # plt.axis('off')
    if title is not None:
        plt.title(title)
    if pfile is not None:
        fig.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(fig)


def display_multiple_images_with_contour(images, segs, subplots=None, window_min=None, window_max=None, slice_number=-1, title=None,
                                         show=True, pfile=None, figsize=None):
    """
    Display multiple image slices with overlay.
    """
    N = len(images)
    assert len(segs) == N, 'Wrong number of segmentations.'
    assert title is None or len(title) == N, \
        'Wrong title. Either title=None or one title per image (which can be None).'
    assert subplots is None or (len(subplots) == 2 and subplots[0] * subplots[1] >= N), "Wrong definition of subplots."

    if figsize is None:
        figsize = (15, 6)

    if subplots is None:
        f, a = plt.subplots(1, N, figsize=figsize)
    else:
        f, a = plt.subplots(subplots[0], subplots[1], figsize=figsize)

    # f, a = plt.subplots(1, N, figsize=(15, 6))

    for n in range(N):
        if images[n] is not None:
            image = check_itk_image(images[n])

            if window_min is None:
                window_min = int(np.amin(image))
            if window_max is None:
                window_max = int(np.amax(image))

            # check image dimension and slice_number
            if len(image.GetSize()) == 2:
                # image is 2D
                img = image
            elif image.GetSize()[2] == 1:
                img = image[:, :, 0]
            else:
                if slice_number < 0 or slice_number > image.GetSize()[2]-1:
                    slice_number = int(image.GetSize()[2]/2)
                img = image[:, :, slice_number]

            if segs[n] is not None:
                seg = check_itk_image(segs[n])
                assert check_image_size(image, seg), 'image and segmentation at pos ' + str(n) + ' should have the same dimensions.'

                if len(seg.GetSize()) == 2:
                    # image is 2D
                    segm = seg
                elif seg.GetSize()[2] == 1:
                    segm = seg[:, :, 0]
                else:
                    segm = seg[:, :, slice_number]
                overlay_img = sitk.LabelMapContourOverlay(sitk.Cast(segm, sitk.sitkLabelUInt8),
                                                          sitk.Cast(sitk.IntensityWindowing(img, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8),
                                                          opacity=1, contourThickness=[2, 2])
                if subplots is None:
                    a[n].imshow(sitk.GetArrayViewFromImage(overlay_img))
                    # a[n].axis('off')
                else:
                    sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
                    a[sub_ind[0], sub_ind[1]].imshow(sitk.GetArrayViewFromImage(overlay_img))
                    # a[sub_ind[0], sub_ind[1]].axis('off')

            else:
                img_np = sitk.GetArrayViewFromImage(img)
                if img_np.shape[0] == 1:
                    img_np = img_np.squeeze(0)
                # We assume the original slice is isotropic, otherwise the display would be distorted
                if subplots is None:
                    # We assume the original slice is isotropic, otherwise the display would be distorted
                    a[n].imshow(img_np, cmap='gray', vmin=window_min, vmax=window_max)
                    # a[n].axis('off')
                else:
                    sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
                    a[sub_ind[0], sub_ind[1]].imshow(img_np, cmap='gray', vmin=window_min, vmax=window_max)
                    # a[sub_ind[0], sub_ind[1]].axis('off')

            if title is not None:
                if title[n] is not None:
                    if subplots is None:
                        a[n].set_title(title[n])
                    else:
                        a[sub_ind[0], sub_ind[1]].set_title(title[n])
            plt.draw()
        else:
            if subplots is not None:
                # a[n].axis('off')
            # else:
                sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
                # a[sub_ind[0], sub_ind[1]].axis('off')

    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)


def display_multiple_images_with_overlay(images, maps, slice_number=-1, title=None,
                                         show=True, pfile=None, subplots=None, figsize=None):
    """
    Display multiple image slices with overlay.
    """
    N = len(images)
    assert len(maps) == N, 'Wrong number of overlay maps.'
    assert title is None or len(title) == N, \
        'Wrong title. Either title=None or one title per image (which can be None).'
    assert subplots is None or (len(subplots) == 2 and subplots[0] * subplots[1] >= N), "Wrong definition of subplots."

    if figsize is None:
        figsize = (15, 6)

    if subplots is None:
        f, a = plt.subplots(1, N, figsize=figsize)
    else:
        f, a = plt.subplots(subplots[0], subplots[1], figsize=figsize)

    for n in range(N):
        if images[n] is not None:
            image = check_itk_image(images[n])

            # check image dimension and slice_number
            if len(image.GetSize()) == 2:
                # image is 2D
                img = image
            elif image.GetSize()[2] == 1:
                img = image[:, :, 0]
            else:
                if slice_number < 0 or slice_number > image.GetSize()[2]-1:
                    slice_number = int(image.GetSize()[2]/2)
                img = image[:, :, slice_number]

            if maps[n] is not None:
                map = check_itk_image(maps[n])
                assert check_image_size(image, map), 'image and segmentation at pos ' + str(n) + ' should have the same dimensions.'

                if len(map.GetSize()) == 2:
                    # image is 2D
                    omap = map
                elif map.GetSize()[2] == 1:
                    omap = map[:, :, 0]
                else:
                    omap = map[:, :, slice_number]

                if subplots is None:
                    a[n].imshow(sitk.GetArrayViewFromImage(img), cmap='gray')
                    divider = make_axes_locatable(a[n])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    a[n].imshow(sitk.GetArrayViewFromImage(omap), cmap=mycmap, alpha=0.3)
                    cim = a[n].imshow(sitk.GetArrayViewFromImage(omap), cmap=mycmap, alpha=0.3)
                    cb0 = f.colorbar(cim, ax=a[n], orientation='vertical', cax=cax)
                    # a[n].axis('off')

                else:
                    sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
                    a[sub_ind[0], sub_ind[1]].imshow(sitk.GetArrayViewFromImage(img), cmap='gray')
                    divider = make_axes_locatable(a[sub_ind[0], sub_ind[1]])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    # a[sub_ind[0], sub_ind[1]].imshow(sitk.GetArrayViewFromImage(omap), cmap=mycmap, alpha=0.7)
                    cim = a[sub_ind[0], sub_ind[1]].imshow(sitk.GetArrayViewFromImage(omap), cmap=mycmap, alpha=0.7)
                    cb0 = f.colorbar(cim, ax=a[sub_ind[0], sub_ind[1]], orientation='vertical', cax=cax)
                    # a[sub_ind[0], sub_ind[1]].axis('off')
            else:
                img_np = sitk.GetArrayViewFromImage(img)
                if img_np.shape[0] == 1:
                    img_np = img_np.squeeze(0)
                # We assume the original slice is isotropic, otherwise the display would be distorted
                if subplots is None:
                    # We assume the original slice is isotropic, otherwise the display would be distorted
                    a[n].imshow(img_np, cmap='gray')
                    # a[n].axis('off')
                else:
                    sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
                    a[sub_ind[0], sub_ind[1]].imshow(img_np, cmap='gray')
                    # a[sub_ind[0], sub_ind[1]].axis('off')

            if title is not None:
                if title[n] is not None:
                    if subplots is None:
                        a[n].set_title(title[n])
                    else:
                        a[sub_ind[0], sub_ind[1]].set_title(title[n])
            plt.draw()
        else:
            if subplots is not None:
                # a[n].axis('off')
            # else:
                sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
                # a[sub_ind[0], sub_ind[1]].axis('off')

    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)


# def display_image_3d(image, title0=None, title1=None, title2=None):
def display_image_3d(image, title=None, show=True, pfile=None):

    """
    Display a CT slice with segmented contours overlaid onto it. The contours are the edges of
    the labeled regions.
    """

    image = check_itk_image(image)

    f, a = plt.subplots(1, 3, figsize=(15, 6))

    slices = [int(image.GetSize()[0]/2), int(image.GetSize()[1]/2), int(image.GetSize()[2]/2)]

    img_X = image[:, :, slices[2]]
    img_Y = image[:, slices[1], :]
    img_Z = image[slices[0], :, :]

    a[0].clear()
    a[0].imshow(sitk.GetArrayViewFromImage(img_X), cmap='gray')
    # a[0].axis('off')
    # if title0 is not None:
    a[0].set_title('X')
    plt.draw()
    a[1].clear()
    a[1].imshow(sitk.GetArrayViewFromImage(img_Y), cmap='gray')
    # a[1].axis('off')
    a[1].set_title('Y')
    plt.draw()
    a[2].clear()
    a[2].imshow(sitk.GetArrayViewFromImage(img_Z), cmap='gray')
    # a[2].axis('off')
    a[2].set_title('Z')
    plt.draw()
    if title is not None:
        f.suptitle(title)

    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)


def display_with_contour_3d(image, segs, window_min=None, window_max=None, title=None, show=True, pfile=None):
    """
    Display a CT slice with segmented contours overlaid onto it. The contours are the edges of
    the labeled regions.
    """

    image = check_itk_image(image)
    segs = check_itk_image(segs)

    if window_min is None:
        window_min = 0
    if window_max is None:
        window_max = 255

    f, a = plt.subplots(1, 3, figsize=(15, 6))

    slices = [int(image.GetSize()[0]/2), int(image.GetSize()[1]/2), int(image.GetSize()[2]/2)]

    img_X = image[:, :, slices[2]]
    msk_X = segs[:, :, slices[2]]
    overlay_X = sitk.LabelMapContourOverlay(sitk.Cast(msk_X, sitk.sitkLabelUInt8),
                                            sitk.Cast(sitk.IntensityWindowing(img_X, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8),
                                            opacity=1, contourThickness=[2, 2])
    img_Y = image[:, slices[1], :]
    msk_Y = segs[:, slices[1], :]
    overlay_Y = sitk.LabelMapContourOverlay(sitk.Cast(msk_Y, sitk.sitkLabelUInt8),
                                            sitk.Cast(sitk.IntensityWindowing(img_Y, windowMinimum=window_min,  windowMaximum=window_max), sitk.sitkUInt8),
                                            opacity=1, contourThickness=[2, 2])
    img_Z = image[slices[0], :, :]
    msk_Z = segs[slices[0], :, :]
    overlay_Z = sitk.LabelMapContourOverlay(sitk.Cast(msk_Z, sitk.sitkLabelUInt8),
                                            sitk.Cast(sitk.IntensityWindowing(img_Z, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8),
                                            opacity=1, contourThickness=[2, 2])

    a[0].clear()
    #a[0].imshow(image[0, 0, :, :, sz].cpu().detach().transpose(0, 1), cmap='gray')
    #a[0].imshow(label[0, 1, :, :, sz].cpu().detach().transpose(0, 1), cmap=mycmap, alpha=0.7)
    a[0].imshow(sitk.GetArrayViewFromImage(overlay_X))
    # a[0].axis('off')
    a[0].set_title('X')
    plt.draw()
    a[1].clear()
    a[1].imshow(sitk.GetArrayViewFromImage(overlay_Y))
    # a[1].axis('off')
    a[1].set_title('Y')
    plt.draw()
    a[2].clear()
    a[2].imshow(sitk.GetArrayViewFromImage(overlay_Z))
    # a[2].axis('off')
    a[2].set_title('Z')
    plt.draw()

    if title is not None:
        f.suptitle(title)

    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)


def display_with_overlay_3d(image, maps, title=None, show=True, pfile=None):
    """
    Display a CT slice with segmented contours overlaid onto it. The contours are the edges of
    the labeled regions.
    """

    image = check_itk_image(image)
    map = check_itk_image(maps)

    f, a = plt.subplots(1, 3, figsize=(15, 6))

    slices = [int(image.GetSize()[0]/2), int(image.GetSize()[1]/2), int(image.GetSize()[2]/2)]

    img_X = image[:, :, slices[2]]
    map_X = map[:, :, slices[2]]

    img_Y = image[:, slices[1], :]
    map_Y = map[:, slices[1], :]

    img_Z = image[slices[0], :, :]
    map_Z = map[slices[0], :, :]

    a[0].clear()
    a[0].imshow(sitk.GetArrayViewFromImage(img_X), cmap='gray')
    a[0].imshow(sitk.GetArrayViewFromImage(map_X), cmap=mycmap, alpha=0.7)
    # a[0].axis('off')
    a[0].set_title('X')
    plt.draw()
    a[1].clear()
    a[1].imshow(sitk.GetArrayViewFromImage(img_Y), cmap='gray')
    a[1].imshow(sitk.GetArrayViewFromImage(map_Y), cmap=mycmap, alpha=0.7)
    # a[1].axis('off')
    a[1].set_title('Y')
    plt.draw()
    a[2].clear()
    a[2].imshow(sitk.GetArrayViewFromImage(img_Z), cmap='gray')
    a[2].imshow(sitk.GetArrayViewFromImage(map_Z), cmap=mycmap, alpha=0.7)
    # a[2].axis('off')
    a[2].set_title('Z')
    plt.draw()

    if title is not None:
        f.suptitle(title)

    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)


def display_multiple_image_3d(images, title=None, show=True, pfile=None):
    """
    Display an image slice.
    """
    N = len(images)
    assert title is None or len(title) == N, \
        'Wrong title. Either title=None or one title per image (which can be None).'
    f, a = plt.subplots(3, N, figsize=(15, 15))

    for n in range(N):
        if images[n] is not None:
            image = check_itk_image(images[n])

            slices = [int(image.GetSize()[0] / 2), int(image.GetSize()[1] / 2), int(image.GetSize()[2] / 2)]

            img_X = image[:, :, slices[2]]
            img_Y = image[:, slices[1], :]
            img_Z = image[slices[0], :, :]

            a[0, n].clear()
            a[0, n].imshow(sitk.GetArrayViewFromImage(img_X), cmap='gray')
            # a[0, n].axis('off')
            if title is not None:
                if title[n] is not None:
                    a[0, n].set_title(title[n])
            plt.draw()
            a[1, n].clear()
            a[1, n].imshow(sitk.GetArrayViewFromImage(img_Y), cmap='gray')
            # a[1, n].axis('off')
            # a[1, n].set_title('Y')
            plt.draw()
            a[2, n].clear()
            a[2, n].imshow(sitk.GetArrayViewFromImage(img_Z), cmap='gray')
            # a[2, n].axis('off')
            # a[2, n].set_title('Z')
            plt.draw()
            plt.draw()

    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)


def display_multiple_image_with_contour_3d(images, segs, window_min=None, window_max=None, title=None, show=True,
                                           pfile=None, figsize=None):
    """
    Display an image slice.
    """
    N = len(images)
    assert len(segs) == N, 'Wrong number of segmentations.'
    assert title is None or len(title) == N, \
        'Wrong title. Either title=None or one title per image (which can be None).'
    if figsize is None:
        f, a = plt.subplots(3, N, figsize=(15, 15))
    else:
        f, a = plt.subplots(3, N, figsize=figsize)

    for n in range(N):
        if images[n] is not None:
            image = check_itk_image(images[n])

            if window_min is None:
                window_min = 0
            if window_max is None:
                window_max = 255

            slices = [int(image.GetSize()[0] / 2), int(image.GetSize()[1] / 2), int(image.GetSize()[2] / 2)]

            img_X = image[:, :, slices[2]]
            img_Y = image[:, slices[1], :]
            img_Z = image[slices[0], :, :]

            if segs[n] is not None:
                seg = check_itk_image(segs[n])
                assert check_image_size(image, seg), 'image and segmentation at pos ' + str(n) + ' should have the same dimensions.'

                seg_X = seg[:, :, slices[2]]
                seg_Y = seg[:, slices[1], :]
                seg_Z = seg[slices[0], :, :]

                overlay_imgX = sitk.LabelMapContourOverlay(sitk.Cast(seg_X, sitk.sitkLabelUInt8),
                                                           sitk.Cast(sitk.IntensityWindowing(img_X, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8),
                                                           opacity=1, contourThickness=[2, 2])
                overlay_imgY = sitk.LabelMapContourOverlay(sitk.Cast(seg_Y, sitk.sitkLabelUInt8),
                                                           sitk.Cast(sitk.IntensityWindowing(img_Y, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8),
                                                           opacity=1, contourThickness=[2, 2])
                overlay_imgZ = sitk.LabelMapContourOverlay(sitk.Cast(seg_Z, sitk.sitkLabelUInt8),
                                                           sitk.Cast(sitk.IntensityWindowing(img_Z, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8),
                                                           opacity=1, contourThickness=[2, 2])

                a[0, n].clear()
                a[0, n].imshow(sitk.GetArrayViewFromImage(overlay_imgX), cmap='gray')
                # a[0, n].axis('off')
                if title is not None:
                    if title[n] is not None:
                        a[0, n].set_title(title[n])
                plt.draw()
                a[1, n].clear()
                a[1, n].imshow(sitk.GetArrayViewFromImage(overlay_imgY), cmap='gray')
                # a[1, n].axis('off')
                # a[1, n].set_title('Y')
                plt.draw()
                a[2, n].clear()
                a[2, n].imshow(sitk.GetArrayViewFromImage(overlay_imgZ), cmap='gray')
                # a[2, n].axis('off')
                # a[2, n].set_title('Z')
                plt.draw()
                plt.draw()
            else:
                a[0, n].clear()
                a[0, n].imshow(sitk.GetArrayViewFromImage(img_X), cmap='gray')
                # a[0, n].axis('off')
                if title is not None:
                    if title[n] is not None:
                        a[0, n].set_title(title[n])
                plt.draw()
                a[1, n].clear()
                a[1, n].imshow(sitk.GetArrayViewFromImage(img_Y), cmap='gray')
                # a[1, n].axis('off')
                # a[1, n].set_title('Y')
                plt.draw()
                a[2, n].clear()
                a[2, n].imshow(sitk.GetArrayViewFromImage(img_Z), cmap='gray')
                # a[2, n].axis('off')
                # a[2, n].set_title('Z')
                plt.draw()
                plt.draw()

    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)



def display_multiple_image_with_overlay_3d(images, maps, title=None, show=True, pfile=None, figsize=None):
    """
    Display an image slice.
    """
    N = len(images)
    assert len(maps) == N, 'Wrong number of segmentations.'
    assert title is None or len(title) == N, \
        'Wrong title. Either title=None or one title per image (which can be None).'
    # f, a = plt.subplots(3, N, figsize=(15, 15))
    if figsize is None:
        f, a = plt.subplots(N, 3, figsize=(15, 15))
    else:
        f, a = plt.subplots(N, 3, figsize=figsize)

    for n in range(N):
        if images[n] is not None:
            image = check_itk_image(images[n])

            slices = [int(image.GetSize()[0] / 2), int(image.GetSize()[1] / 2), int(image.GetSize()[2] / 2)]

            img_X = image[:, :, slices[2]]
            img_Y = image[:, slices[1], :]
            img_Z = image[slices[0], :, :]

            a[n, 0].clear()
            a[n, 0].imshow(sitk.GetArrayViewFromImage(img_X), cmap='gray')
            a[n, 1].clear()
            a[n, 1].imshow(sitk.GetArrayViewFromImage(img_Y), cmap='gray')
            a[n, 2].clear()
            a[n, 2].imshow(sitk.GetArrayViewFromImage(img_Z), cmap='gray')

            if not isinstance(maps[n], list):

                if maps[n] is not None:
                    map = check_itk_image(maps[n])
                    assert check_image_size(image, map), 'image and segmentation at pos ' + str(n) + ' should have the same dimensions.'

                    map_X = map[:, :, slices[2]]
                    map_Y = map[:, slices[1], :]
                    map_Z = map[slices[0], :, :]

                
                if maps[n] is not None:
                    a[n, 0].imshow(sitk.GetArrayViewFromImage(map_X), cmap=mycmap, alpha=0.7)
                    a[n, 1].imshow(sitk.GetArrayViewFromImage(map_Y), cmap=mycmap, alpha=0.7)
                    a[n, 2].imshow(sitk.GetArrayViewFromImage(map_Z), cmap=mycmap, alpha=0.7)
            else:
                for m in range(len(maps[n])):
                    map = check_itk_image(maps[n][m])
                    assert check_image_size(image, map), 'image and segmentation at pos ' + str(n) + ' should have the same dimensions.'

                    map_X = map[:, :, slices[2]]
                    map_Y = map[:, slices[1], :]
                    map_Z = map[slices[0], :, :]

                    a[n, 0].imshow(sitk.GetArrayViewFromImage(map_X), cmap=mycmaps[m], alpha=0.7-m*0.4)
                    a[n, 1].imshow(sitk.GetArrayViewFromImage(map_Y), cmap=mycmaps[m], alpha=0.7-m*0.4)
                    a[n, 2].imshow(sitk.GetArrayViewFromImage(map_Z), cmap=mycmaps[m], alpha=0.7-m*0.4)


            if title is not None:
                if title[n] is not None:
                    a[n, 0].set_title(title[n])


    if pfile is not None:
        f.savefig(pfile)
    if show:
        plt.show()
    else:
        plt.close(f)