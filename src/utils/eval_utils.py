import os
import sys
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn import metrics
import torch


def evaluation(classes, ground_truth, prediction, confidences, sheet_name, outfile, file_names):
    C = len(classes)
    N = len(ground_truth)

    # print predictions and labels
    measures = ['gt', 'pred']
    for c in range(0, C):
        measures.append('class {}: {}'.format(c, classes[c]))
    measures += ['classes', 'BAcc', 'Sens', 'Spec', 'F1', 'Prec']
    values = np.empty((N, len(measures)))
    values[:] = np.nan

    values[:, 0] = ground_truth
    values[:, 1] = prediction
    for n in range(0, N):
        for c in range(0, C):
            values[n, c + 2] = confidences[n][c]

    # print overall and class specific performance scores in separate sheet
    bacc, sens, spec, f1, prec = performance_scores(ground_truth, prediction)

    values[1:C+1, C+2] = range(0, C)
    values[0, C+3] = bacc
    values[0, C+4] = sens
    values[0, C+5] = spec
    values[0, C+6] = f1
    values[0, C + 7] = prec

    # class specific:
    for c in range(0, C):
        gt_c = np.where(np.array(ground_truth) == c, 1, 0)
        pred_c = np.where(np.array(prediction) == c, 1, 0)
        bacc_c, sens_c, spec_c, f1_c, prec_c = performance_scores(gt_c, pred_c)
        values[c+1, C + 3] = bacc_c
        values[c+1, C + 4] = sens_c
        values[c+1, C + 5] = spec_c
        values[c+1, C + 6] = f1_c
        values[c + 1, C + 7] = prec_c

    save_to_excel(file_names, values, measures, sheet_name, outfile, classes)


def save_to_excel(names, values, measures, sheet_name, outfile, classes, task='classify', **kwargs):
    # save volumes in excel file
    if os.path.isfile(outfile):
        book = load_workbook(outfile)
        writer = pd.ExcelWriter(outfile, engine='openpyxl')
        writer.book = book

    N, M = values.shape

    if N > 1:
        d = {'ids': range(0, N)}
    else:
        d = {'ids': range(0, N - 2)}
    df3 = pd.DataFrame(data=d)

    # save to .xlsx file
    df3['name'] = names
    if task=='segment':
        if classes is not None:
            df3['classes'] = classes

    # additional columns
    if len(kwargs)>0:
        for k,v in kwargs.items():
            df3[k] = v

    if task=='segment':
        for m in range(0, len(measures)):
            for j in range(0, N):
                if np.isinf(values[j, m]):
                    values[j, m] = -1
            df3[measures[m]] = values[:, m].tolist()
    else:
        for m in range(0, len(measures)):
            df3[measures[m]] = values[:, m].tolist()
        df3['class names'] = ['overall'] + classes + (len(names)-(len(classes)+1))*[' ']

    if os.path.isfile(outfile):
        df3.to_excel(writer, sheet_name=sheet_name)
        writer.save()
        writer.close()
    else:
        df3.to_excel(outfile, sheet_name=sheet_name)


def performance_scores(gt, pred):


    # #
    # # Accuracy
    # #
    # d = np.trace(cfm)
    # acc = d / len(gt)
    # #
    # # Balanced Accuracy
    # #
    # d = 0
    # for i in range(0, NC):
    #     d = d + cfm[i, i] / np.sum(cfm[i, :])
    # bacc = d / NC

    specificity = specificity_score(gt, pred)
    accuracy = metrics.balanced_accuracy_score(gt, pred)
    f1 = metrics.f1_score(gt, pred, average='macro')
    sensitivity = metrics.recall_score(gt, pred, average='macro')
    precision = metrics.precision_score(gt, pred, average='macro')

    return accuracy, sensitivity, specificity, f1, precision


def specificity_score(gt, pred):
    confusion_matrix = metrics.confusion_matrix(gt, pred)
    confusion_matrix = confusion_matrix.astype(float)
    NC = confusion_matrix.shape[0]

    specificity = -1
    if NC==2:
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FN = confusion_matrix[1, 0]
        FP = confusion_matrix[0, 1]
        if (float(FP) + float(TN)) > 0:
            specificity = float(TN) / (float(FP) + float(TN))
    elif NC>2:
        # number of classes:
        c = len(confusion_matrix)
        # print c
        cfmat = np.zeros((c, c))
        for i in range(0, c):
            for j in range(0, c):
                cfmat[i, j] = confusion_matrix[i, j]
        # print cfmat

        spec = -1*np.ones(c)

        for i in range(0, c):
            # print
            FP = np.sum(cfmat[:, i]) - cfmat[i, i]

            cfm = np.delete(cfmat, i, 0)
            cfm = np.delete(cfm, i, 1)
            TN = cfm.sum()

            if (float(FP) + float(TN)) > 0:
                spec[i] = float(TN) / (float(FP) + float(TN))
        # spec[i] = cfmat[i,i]/np.sum(cfmat[i,:])

        specificity = np.mean(spec)
    return specificity

def dice(pred, gt, class_labels):
    """
    Arguments
    ---------
    pred : Tensor
        of size batch_size X 1 X (img_size).
        Predicted segmentation.
    gt : Tensor
        of size batch_size X 1 X (img_size).
        Ground truth segmentation.
    class_labels : sequence of int
        label values of the classes for which to compute dice scores.
        These are the values potentially present in gt.

    Returns
    -------
    dice_score : sequence of float with length = len(class_labels)
        Dice scores for each class represented by the elements of class_labels.
        If class_labels[idx] is not present in both pred and gt, dice score is set to 1.
    """
    dice_scores = torch.zeros(len(class_labels))
    batch_size = pred.size()[0]
    # View gt and pred as batch_size X num_of_voxels i.e. one row is one training sample
    gt = gt.contiguous().view(batch_size, -1)
    pred = pred.contiguous().view(batch_size, -1)
    for idx, label in enumerate(class_labels):
        # Compute unions and intersections for each sample in this batch
        unions = (pred == label).float().sum(1) + (gt == label).float().sum(1)  # Now of size batch_size
        # Float conversion needed because: https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
        intersections = ((pred == label) * (gt == label)).float().sum(1)  # Now of size batch_size
        # Where unions==0, modify s.t. we'll have dice score of 1
        intersections[unions <= 0] = 1.
        unions[unions <= 0] = 2.
        # Take mean over the samples in this batch, and hence get a scalar
        _dice = (2. * intersections / unions).mean()
        dice_scores[idx] = _dice  # Add dice score to the corresponding label index
    return dice_scores


def evaluate_segmentation(reference, segmentation):

    # to sitk image
    gt = sitk.GetImageFromArray(reference)
    pred = sitk.GetImageFromArray(segmentation)

    iou, dice = get_overlap_measures(gt, pred)
    asd, hd95 = get_surface_distance_measures(gt, pred)
    
    return iou, dice, asd, hd95


def get_overlap_measures(reference, segmentation):

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(segmentation, reference)

    return overlap_measures_filter.GetJaccardCoefficient(), overlap_measures_filter.GetDiceCoefficient()


def get_surface_distance_measures(reference, segmentation):

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    label = 1
    statistics_image_filter = sitk.StatisticsImageFilter()

    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(reference, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(reference)
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    # Hausdorff distance
    try:
        hausdorff_distance_filter.Execute(reference, segmentation)
        hausdorff = hausdorff_distance_filter.GetHausdorffDistance()
    except RuntimeError:
        hausdorff = np.nan

    # surface distances
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(segmentation, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(segmentation)

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get the number of pixels in the segmented surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

    all_surface_distances = seg2ref_distances + ref2seg_distances

    # Robust hausdorff distance: X% percentile of
    robust_hausdorff_95 = np.percentile(all_surface_distances, 95)

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
    # segmentations, though in our case it is. More on this below.
    return np.mean(all_surface_distances), robust_hausdorff_95
    # return hausdorff, 0., 0., 0., 0.