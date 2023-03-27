# Author: Veronika Zimmer <vam.zimmer@gmail.com, veronika.zimmer@tum.de>
# 	  King's College London, UK
#     TU Munich, Germany

"""
Dataset loader for US placenta imaging dataset with input images and ground truth
segmentation.
Samples are read from excel sheets.

Available functions:
    isNaN()
    PlacentaT2Dataset() :   Segmentation loader
    PlacentaT1Dataset() :   Classification loader
    PlacentaMTDataset() :   Multitask loader


"""
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import torchio as tio

from src.dataset import data_utils


def isNaN(num):
    return num != num


class PlacentaDataset(Dataset):
    def __init__(self, root, data_file, sheet_name='images', mode='train',
                 classes=None,
                 train_transform=None, val_transform=None):
        # training set or test set
        assert(mode in ['train', 'validate', 'test']), 'Please provide a valid mode.'
        self.mode = mode
        self.root = root
        self.sheet_name = sheet_name
        self.classes_in = classes
        self.number_images = -1

        self.images_root = self.root
        self.labels_root = None
        self.classes = ['anterior', 'none', 'posterior']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        

        # for each patient the images with corresponding information (mux info and parameters, placental position etc)
        # is stored in data_file
        assert os.path.isfile(data_file), 'Please provide a valid data file. \n (not valid {})'.format(data_file)
        self.data_file = data_file

        self.train_transform = train_transform
        self.val_transform = val_transform

    def _read_sheet(self, segm=False):
        df = pd.read_excel(self.data_file, sheet_name=self.sheet_name)

        df_mode = df.loc[df['set']==self.mode]
        if segm:
            df_mode = df_mode.loc[df['segmentation']==1]

        self.filenames = [f'{f}.mhd' for f in list(df_mode['image_name'])]
        self.class_labels = list(df_mode['position'])
        self.segmentations = list(df_mode['segmentation'])

        self.number_images = len(self.filenames)

        return df_mode
    
    def _prepare_classes(self):

        self.class_cardinality = np.asarray([self.class_labels.count(i) for i in self.classes])
        print(self.class_cardinality)
        for c in range(len(self.classes)):
            print("Class {} ({}):\t\t {} images ({:1.2f}%)".format(c, self.classes[c], self.class_cardinality[c],
                                                                   100 * self.class_cardinality[c] / len(self.filenames)))

        prob = self.class_cardinality / float(np.sum(self.class_cardinality))
        classes = [self.class_to_idx[l] for l in self.class_labels]
        reciprocal_weights = [prob[classes[index]] for index in range(len(classes))]
        weights = (1. / np.array(reciprocal_weights))
        self.sample_weights = weights / np.sum(weights)

    def _check_images(self):
        for f in self.filenames:
            # if not os.path.exists("{}/{}".format(self.images_root, f)):
            if not os.path.exists(f"{self.images_root}/{f}"):
                # print("Image: " + f)
                print(f"Image: {self.images_root}/{f}")

    def _check_labels(self):
        for f in self.filenames:
            if not os.path.exists("{}/{}".format(self.labels_root, f.replace('.mhd', '.mha'))):
                print("Label: " + f.replace('.mhd', '.mha'))

    def __len__(self):
        return len(self.filenames)

    def get_filenames(self):
        return self.filenames

    def get_classes(self):
        return self.classes

    def get_class_cardinality(self):
        return self.class_cardinality

    def get_sample_weights(self):
        return self.sample_weights


class PlacentaT2Dataset(PlacentaDataset):
    def __init__(self, root, data_file, labels_root, sheet_name='images', mode='train',
                 train_transform=None, val_transform=None):
        super(PlacentaT2Dataset, self).__init__(root, data_file,
                                                sheet_name=sheet_name, mode=mode,
                                                train_transform=train_transform, val_transform=val_transform)

        self.labels_root = labels_root

        df = self._read_sheet(segm=True)

        self._check_images()
        self._check_labels()

    def __getitem__(self, index):
        """
        Arguments
        ---------
        index : int
            index position to return the data
        Returns
        -------
         tuple: (image, label, info) where 
                    label is the ground truth segmentation, 
                    info is the image id and placenta position as string.
        """
        fname = self.filenames[index]
        fname_file = f"{self.images_root}/{fname}"
        lname_file = f"{self.labels_root}/{fname.replace('.mhd','.mha')}"
        name = os.path.splitext(fname)[0]

        # clabel = np.empty([1])

        class_label = self.class_labels[index]
        # clabel = [0] * len(self.classes)
        # clabel[self.class_to_idx[class_label]] = 1
        # clabel = np.array(clabel, dtype=np.float32)

        image = {
            'Image': tio.ScalarImage(fname_file),
            'Label': tio.LabelMap(lname_file)
        }
        subject = tio.Subject(image)

        if self.train_transform is not None and self.mode=='train':
            # print('Train data augmentation')
            subject = self.train_transform(subject)
        if self.val_transform is not None and not self.mode=='train':
            # print('Val/Inf data augmentation')
            subject = self.val_transform(subject)

        images, label = data_utils.create_tensor_from_torchio_subject(subject, image_names=['Image'], mask_names=['Label'])
        # print(images.size())


        return images, label, [name, class_label]

class PlacentaT1Dataset(PlacentaDataset):
    def __init__(self, root, data_file, sheet_name='images', mode='train', 
                 train_transform=None, val_transform=None):
        super(PlacentaT1Dataset, self).__init__(root, data_file,
                                                sheet_name=sheet_name, mode=mode,
                                                train_transform=train_transform, val_transform=val_transform)

        df = self._read_sheet()
        self._prepare_classes()
        self._check_images()

    def __getitem__(self, index):
        """
        Arguments
        ---------
        index : int
            index position to return the data
        Returns
        -------
         tuple: (image, clabel, info) where 
                    clabel the one-hot encoded classification label  (placenta position)
                    info is the image id and placenta position as string.
        """
        fname = self.filenames[index]
        label = None
        clabel = np.empty([1])

        class_label = self.class_labels[index]
        clabel = [0] * len(self.classes)
        clabel[self.class_to_idx[class_label]] = 1
        clabel = np.array(clabel, dtype=np.float32)

        fname_file = '{}/{}'.format(self.images_root, fname)
        name = os.path.splitext(fname)[0]

        image = {
            'Image': tio.ScalarImage(fname_file),
        }
        subject = tio.Subject(image)

        if self.train_transform is not None and self.mode=='train':
            subject = self.train_transform(subject)
        if self.val_transform is not None and not self.mode=='train':
            subject = self.val_transform(subject)

        images, _ = data_utils.create_tensor_from_torchio_subject(subject, image_names=['Image'])

        return images, clabel, name


class PlacentaMTDataset(PlacentaDataset):
    def __init__(self, root, data_file, labels_root, sheet_name='images', mode='train',
                 train_transform=None, val_transform=None):
        super(PlacentaMTDataset, self).__init__(root, data_file,
                                                sheet_name=sheet_name, mode=mode,
                                                train_transform=train_transform, val_transform=val_transform)

        self.labels_root = labels_root

        df = self._read_sheet(segm=True)

        self._prepare_classes()
        self._check_images()
        self._check_labels()

    def __getitem__(self, index):
        """
        Arguments
        ---------
        index : int
            index position to return the data
        Returns
        -------
        tuple: (image, label, clabel, info) where 
                    label is the ground truth segmentation, 
                    clabel the one-hot encoded classification label  (placenta position)
                    info is the image id and placenta position as string.
        """
        fname = self.filenames[index]
        fname_file = f"{self.images_root}/{fname}"
        lname_file = f"{self.labels_root}/{fname.replace('.mhd','.mha')}"
        name = os.path.splitext(fname)[0]

        class_label = self.class_labels[index]
        clabel = [0] * len(self.classes)
        clabel[self.class_to_idx[class_label]] = 1
        clabel = np.array(clabel, dtype=np.float32)

        image = {
            'Image': tio.ScalarImage(fname_file),
            'Label': tio.LabelMap(lname_file)
        }
        subject = tio.Subject(image)

        if self.train_transform is not None and self.mode=='train':
            subject = self.train_transform(subject)
        if self.val_transform is not None and not self.mode=='train':
            subject = self.val_transform(subject)

        images, label = data_utils.create_tensor_from_torchio_subject(subject, image_names=['Image'], mask_names=['Label'])

        return images, label, clabel, [name, class_label]