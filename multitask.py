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
from src.dataset.placenta_data import PlacentaT1Dataset, PlacentaT2Dataset, PlacentaMTDataset
from src.utils import python_utils as putils
from src.utils import display_utils
from src.worker import multitasker

# import class_exp10 as config
config_file = sys.argv[1].split(".")[0]  # remove ending '.py'
config = __import__(config_file)
config.MODE = sys.argv[2]

# --------------------------------------------------------------------

print(putils.bcolors.OKGREEN + "Multitask (Segmentation + Classification) of placentas..." + putils.bcolors.ENDC)

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
config.RESUME = config.OUT_DIR + '/' + config.EXPERIMENT_PREFIX + 'checkpoint.pth.tar'
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

train_t2_loader = None
train_t1_loader = None
val_loader = None
subsets_infer = [('anterior', 'posterior')]
inf_loader = None
if config.MODE == 'train':
    print("TRAINING DATA")
    train_t2_dataset = PlacentaT2Dataset(DATA_DIR, DATA_FILE, LABEL_DIR,
                                      sheet_name=SHEET_NAME,
                                      mode='train',
                                      train_transform=train_tfm, val_transform=val_tfm
                                      )

    train_t2_loader = DataLoader(train_t2_dataset, num_workers=config.WORKERS, shuffle=True,
                              batch_size=1, pin_memory=False)  # pin_memory=True)
    print('[Segmentation] Train loader size = {}'.format(len(train_t2_loader)))
    image2, label2, info2 = train_t2_dataset[0]

    train_t1_dataset = PlacentaT1Dataset(DATA_DIR, DATA_FILE, 
                                      sheet_name=SHEET_NAME,
                                      mode='train',
                                      train_transform=train_tfm, val_transform=val_tfm
                                      )

    train_t1_loader = DataLoader(train_t1_dataset, num_workers=config.WORKERS, shuffle=True,
                              batch_size=1, pin_memory=False)  # pin_memory=True)
    print('[Classification] Train loader size = {}'.format(len(train_t1_loader)))
    image1, label1, info1 = train_t1_dataset[0]

    print("VALIDATION DATA")
    val_batch_size = config.BATCH_SIZE
    val_dataset = PlacentaMTDataset(DATA_DIR, DATA_FILE, LABEL_DIR,
                                    sheet_name=SHEET_NAME,
                                    mode='validate', 
                                    train_transform=train_tfm, val_transform=val_tfm
                                    )
    val_loader = DataLoader(val_dataset, num_workers=config.WORKERS, shuffle=False,
                            batch_size=val_batch_size, pin_memory=False)  # pin_memory=True)
    print('[Multitask] Validate loader size = {}'.format(len(val_loader)))
    valimage, vallabel, valclabel, info = val_dataset[0]

    # Loss function
    n_samples_per_class = train_t1_dataset.get_class_cardinality()
    class_weights = sum(n_samples_per_class) / n_samples_per_class
    for i in range(len(config.CLASSES)):
        print('#Samples of class %d: %d, weight: %f ' % (i, n_samples_per_class[i], class_weights[i]))
    print()
    config.criterionT1 = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights)).type(torch.FloatTensor).cuda()


elif 'infer' in config.MODE:
    inf_loader = []
    inf_dataset = PlacentaMTDataset(DATA_DIR, DATA_FILE, LABEL_DIR,
                                    sheet_name=SHEET_NAME,
                                    mode='test', 
                                    train_transform=train_tfm, val_transform=val_tfm
                                    )
    inf_loader = DataLoader(inf_dataset, num_workers=config.WORKERS, shuffle=False,
                                 batch_size=config.BATCH_SIZE_INF, pin_memory=False)  # pin_memory=True)

    print('[Multitask] Inference loader size = {}'.format(len(inf_loader)))
    print()

"""
    Training
"""
multitasking = multitasker.Multitasker(config, train_t1_loader, train_t2_loader, val_loader, inf_loader)
if config.MODE == 'train':
    print(putils.bcolors.OKGREEN + "Training..." + putils.bcolors.ENDC)
    # print(config.model)
    multitasking.train()

elif 'infer' in config.MODE:
    print(putils.bcolors.OKGREEN + "Testing..." + putils.bcolors.ENDC)

    # inference
    if config.MODE == 'infer' or config.DROPOUT_MC==0:
        multitasking.infer()
    elif config.MODE == 'infer_mc' and config.DROPOUT_MC>0:
        multitasking.infer_mc_dropout()
    else:
        print("Check infer mode or mc dropout.")