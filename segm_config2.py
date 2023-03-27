import torch
import torch.optim as optim
from src.losses.losses import DiceBCELoss
from src.models.unet3d import ResUNet3D as Unet
import random

PREFIX = 'segm_exp2'
PRETRAINED = 'class_exp1'
TASK = 'segment'
FREEZE_ENCODER = False

EPOCHS = 70
START_EPOCH = 0
BATCH_SIZE = 1
BATCH_SIZE_INF = 1
OPTIM = 'Adam'  # 'SGD'
# Learning rates
SCHEDULER = 'MultiStepLR'
LEARNING_RATE = 0.0001
GAMMA = 0.1                 # factor for reducing the learning rates
STEP = 30                   # for StepLR
MILESTONES = [30, 70, 90]   # for MultiLR
# L2_REGULARIZER = 1e-5
MOMENTUM = 0.99
WEIGHT_DECAY = 5e-4         # L2 regularization
LOSS_TYPE = 'Hybrid'
LOSS_WEIGHT = 0.5

# unet architecure
LEVEL = 5
NCHANNELS = 16
DROPOUT = False
DROPOUT_DEC = True
DROPOUT_RATE = 0.2
DROPOUT_MC = 3
CONCATENATE = False
TIME = 2

PLOT_PROGRESS = True
EVALUATE = True

WORKERS = 4
PRINT_FREQ = 5

# data transform and augmentation
USE_WHITENING = True
RESCALING = [0, 1]
RESAMPLE_SIZE = [128, 128, 128]
CROPPAD = False
BLUR = [False, 0.5]
NOISE = [False, 0.5]
SPATIAL = [1.0, 1.0, 0]
FLIP = [0.5, ('LR','IS')]

# -------- Model + Loss ----------------------------------------------------------

OUT_CHANNELS = 1
criterion = DiceBCELoss()
print('hybrid (dice + cross entropy) loss with {} out-channels'.format(OUT_CHANNELS))

SEED = 42
torch.manual_seed(SEED)

model = Unet(inchannels=1, outchannels=OUT_CHANNELS, first_channels=NCHANNELS,
             image_size=tuple(RESAMPLE_SIZE), levels=LEVEL,
             dropout=DROPOUT, dropout_rate=DROPOUT_RATE, dropout_dec=DROPOUT_DEC,
             concatenation=CONCATENATE)

# -------- Optimizer + Scheduler ----------------------------------------------------------

if torch.cuda.is_available():
    print('CUDA')
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)
print('optimizer lr = {}'.format(LEARNING_RATE))
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
