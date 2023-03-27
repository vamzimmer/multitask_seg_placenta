import os
import sys
import torch
from torch.utils.data import DataLoader

DATA_FILE = 'datainfo.xlsx'
DATA_DIR = "../images"
LABEL_DIR = '../labels'
OUT_FOL = "../placenta_project"
SCRIPT_DIR = "../multitask_seg_placenta"

from src.dataset import data_utils
from src.dataset.placenta_data import PlacentaT2Dataset as PlacentaT2Dataset
from src.utils import python_utils as putils
from src.utils import display_utils
from src.worker import segmenter

# import class_exp10 as config
config_file = sys.argv[1].split(".")[0]  # remove ending '.py'
config = __import__(config_file)

config.MODE = sys.argv[2]

# --------------------------------------------------------------------

print(putils.bcolors.OKGREEN + "Segmentation of placentas..." + putils.bcolors.ENDC)

config.SHEET = ''
SHEET_NAME = 'images_sort'

prefix = config.PREFIX
config.OUT_DIR = OUT_FOL + "/" + prefix
config.EXPERIMENT_PREFIX = prefix
if not os.path.exists(config.OUT_DIR):
    os.makedirs(config.OUT_DIR)
config.PROGRESS_FILE = f"{SCRIPT_DIR}/{prefix}.png"
config.RESULTS_FILE = f"{SCRIPT_DIR}/{prefix}_results.xlsx"

# get model filename for the pretrained classification model
if config.PRETRAINED is not None:
    config.PRETRAINED = f'{OUT_FOL}/{config.PRETRAINED}/{config.PRETRAINED}model_best_loss.pth.tar'

config.RESUME = False
# config.RESUME = config.OUT_DIR + '/' + config.EXPERIMENT_PREFIX + 'checkpoint.pth.tar'
if 'infer' in config.MODE:
    config.RESUME = config.OUT_DIR + '/' + config.EXPERIMENT_PREFIX + 'model_best.pth.tar'

    # if we want to test only on the pretrained classification model without further multitask training
    # config.RESUME = config.OUT_DIR + '/' + config.EXPERIMENT_PREFIX + 'checkpoint_pretrained.pth.tar'


# ----------------------------------------------------------------------------------

"""
    Prepare the data
"""
optn = {}
optn["whitening"] = config.USE_WHITENING
optn["rescaling"] = config.RESCALING
optn["resample_size"] = config.RESAMPLE_SIZE
optn["croppad"] = config.CROPPAD
optn["blur"] = config.BLUR[0]
optn["blur_prob"] = config.BLUR[1]
optn["noise"] = config.NOISE[0]
optn["noise_prob"] = config.NOISE[1]
optn["spatial_prob"] = config.SPATIAL[0]
optn["affine_prob"] = config.SPATIAL[1]
optn["elastic_prob"] = config.SPATIAL[2]
optn["flip_prob"] = config.FLIP[0]
optn["flip_axes"] = config.FLIP[1]
train_tfm, val_tfm = data_utils.data_transforms(**optn)

train_loader = None
val_loader = None
subsets_infer = [('anterior', 'posterior')]
inf_loader = None
if config.MODE == 'train':
    print("TRAINING DATA")
    train_dataset = PlacentaT2Dataset(DATA_DIR, DATA_FILE, LABEL_DIR,
                                      sheet_name=SHEET_NAME,
                                      mode='train',
                                      train_transform=train_tfm, val_transform=val_tfm
                                      )

    train_loader = DataLoader(train_dataset, num_workers=config.WORKERS, shuffle=True,
                              batch_size=1, pin_memory=False)  # pin_memory=True)
    print('Train loader size = {}'.format(len(train_loader)))
    image, label, info = train_dataset[0]

    print("VALIDATION DATA")
    val_batch_size = config.BATCH_SIZE
    val_dataset = PlacentaT2Dataset(DATA_DIR, DATA_FILE, LABEL_DIR,
                                    sheet_name=SHEET_NAME,
                                    mode='validate', 
                                    train_transform=train_tfm, val_transform=val_tfm
                                    )
    val_loader = DataLoader(val_dataset, num_workers=config.WORKERS, shuffle=False,
                            batch_size=val_batch_size, pin_memory=False)  # pin_memory=True)
    print('Validate loader size = {}'.format(len(val_loader)))
    valimage, vallabel, info = val_dataset[0]

    display_utils.display_3d([image[0,:], valimage[0,:]], [label[0,:], vallabel[0,:]], pfile='/home/veronika/Code/muxtk/pubrepo/sdata.png')

elif 'infer' in config.MODE:
    inf_loader = []
    inf_dataset = PlacentaT2Dataset(DATA_DIR, DATA_FILE, LABEL_DIR,
                                    sheet_name=SHEET_NAME,
                                    mode='test', 
                                    train_transform=train_tfm, val_transform=val_tfm
                                    )
    inf_loader = DataLoader(inf_dataset, num_workers=config.WORKERS, shuffle=False,
                                 batch_size=config.BATCH_SIZE_INF, pin_memory=False)  # pin_memory=True)

    print('Inference loader size = {}'.format(len(inf_loader)))
    print()

"""
    Training
"""
segmenting = segmenter.Segmenter(config, train_loader, val_loader, inf_loader)
if config.MODE == 'train':
    print(putils.bcolors.OKGREEN + "Training..." + putils.bcolors.ENDC)
    # print(config.model)
    segmenting.train()

elif 'infer' in config.MODE:
    print(putils.bcolors.OKGREEN + "Testing..." + putils.bcolors.ENDC)

    config.SHEET = ''
    # inference
    if config.MODE == 'infer' or config.DROPOUT_MC==0:
        segmenting.infer()
    elif config.MODE == 'infer_mc' and config.DROPOUT_MC>0:
        segmenting.infer_mc_dropout()
    else:
        print("Check infer mode or mc dropout.")