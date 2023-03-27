import os
import sys
import torch
from torch.utils.data import DataLoader

DATA_FILE = '/home/veronika/Code/muxtk/pubrepo/datainfo.xlsx'
DATA_DIR = "/media/veronika/Veronika8TB/iFIND/Placenta/Data/single/images"
OUT_FOL = "/home/veronika/Out/test_placenta_project"
SCRIPT_DIR = "/home/veronika/Code/muxtk/pubrepo"

sys.path.insert(0, '/home/veronika/Code/muxtk')
from src.dataset import data_utils
from src.dataset.placenta_data import PlacentaT1Dataset as PlacentaT1Dataset
from src.utils import python_utils as putils
from src.worker import classifier

config_file = sys.argv[1].split(".")[0]  # remove ending '.py'
config = __import__(config_file)

config.MODE = sys.argv[2]

# --------------------------------------------------------------------

print(putils.bcolors.OKGREEN + "Classification of anterior/posterior placentas..." + putils.bcolors.ENDC)

subset_name = 'position'  
used_classes = ['anterior', 'none', 'posterior']

config.SHEET = ''
SHEET_NAME = 'images_sort'


prefix = config.PREFIX
config.OUT_DIR = OUT_FOL + "/" + prefix
config.EXPERIMENT_PREFIX = prefix
if not os.path.exists(config.OUT_DIR):
    os.makedirs(config.OUT_DIR)
config.PROGRESS_FILE = f"{SCRIPT_DIR}/{prefix}.png"
config.RESULTS_FILE = f"{SCRIPT_DIR}/{prefix}_results.xlsx"

config.RESUME = False
config.RESUME = config.OUT_DIR + '/' + config.EXPERIMENT_PREFIX + 'checkpoint.pth.tar'
if config.MODE == 'infer':
    config.RESUME = config.OUT_DIR + '/' + config.EXPERIMENT_PREFIX + 'model_best.pth.tar'
    config.DROPOUT = False

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
    train_dataset = PlacentaT1Dataset(DATA_DIR, DATA_FILE, 
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
    val_dataset = PlacentaT1Dataset(DATA_DIR, DATA_FILE,
                                    sheet_name=SHEET_NAME,
                                    mode='validate', 
                                    train_transform=train_tfm, val_transform=val_tfm
                                    )
    val_loader = DataLoader(val_dataset, num_workers=config.WORKERS, shuffle=False,
                            batch_size=val_batch_size, pin_memory=False)  # pin_memory=True)
    print('Validate loader size = {}'.format(len(val_loader)))
    valimage, vallabel, info = val_dataset[0]

    # display_utils.display_3d([image[0,:], valimage[0,:]], pfile='/home/veronika/Code/muxtk/pubrepo/data.png')

    # Loss function
    n_samples_per_class = train_dataset.get_class_cardinality()
    class_weights = sum(n_samples_per_class) / n_samples_per_class
    for i in range(len(config.CLASSES)):
        print('#Samples of class %d: %d, weight: %f ' % (i, n_samples_per_class[i], class_weights[i]))
    print()
    config.criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights)).type(torch.FloatTensor).cuda()


elif config.MODE == 'infer':
    inf_loader = []
    inf_dataset = PlacentaT1Dataset(DATA_DIR, DATA_FILE, 
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
classifying = classifier.Classifier(config, train_loader, val_loader, inf_loader)
if config.MODE == 'train':
    print(putils.bcolors.OKGREEN + "Training..." + putils.bcolors.ENDC)
    # print(config.model)
    classifying.train()

elif config.MODE == 'infer':
    print(putils.bcolors.OKGREEN + "Testing..." + putils.bcolors.ENDC)

    classifying = classifier.Classifier(config, train_loader, val_loader, inf_loader)
    config.SHEET = ''
    # inference
    classifying.infer()