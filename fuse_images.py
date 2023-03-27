import os
import sys
import SimpleITK as sitk

DATA_DIR = "../images"
OUT_FOL = "../placenta_project/fusion"
SCRIPT_DIR = "../multitask_seg_placenta"

if not os.path.exists(OUT_FOL):
    os.makedirs(OUT_FOL)

from src.utils import display_utils
from src.fusion import fuse_utils as futils

number_images = int(sys.argv[1])
fusion_method = sys.argv[2] 

# variables for the fusion
optn = {}
optn['fusion_method'] = fusion_method # {'alignment', 'maximum', 'average', 'addition', 'frustum'}

if number_images==3:
    image_names = ['image_name_1', 'image_name_2', 'image_name_3']
    output_file = f'{OUT_FOL}/fused-3.mhd'
    optn['flip'] = [1, 1, 1]      # 1: rotate images by 180 deg, if necessary due to probe orientation in the holder
    optn['default_transform'] = 2 # for three images, calibrated to our US probe holder 
else:
    image_names = ['image_name_1', 'image_name_2']
    output_file = f'{OUT_FOL}/fused-2.mhd'
    optn['flip'] = [0, 1]         # # 1: rotate images by 180 deg, if necessary due to probe orientation in the holder
    optn['default_transform'] = 1 # for two images, calibrated to our US probe holder 

image_files = [f'{DATA_DIR}/{ifile}.mhd' for ifile in image_names]

images = [sitk.ReadImage(img) for img in image_files]

for img in images:
    print(img.GetSpacing(), img.GetSize())

futils.fuse_images(image_files, output_file, **optn)
