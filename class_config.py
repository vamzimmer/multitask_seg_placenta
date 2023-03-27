import torch
import torch.optim as optim
from src.models.encoder3d import ResEncoderAtt as Net
import random

PREFIX = 'class_exp1'
USED_CLASSES = 'Default'
TASK = 'classify'

EPOCHS = 20
START_EPOCH = 0
BATCH_SIZE = 1
BATCH_SIZE_INF = 2
OPTIM = 'Adam'  # 'SGD'
# Learning rates
SCHEDULER = 'None'  # 'None'
LEARNING_RATE = 0.00001
GAMMA = 0.1                 # factor for reducing the learning rates
STEP = 20                  # for StepLR
# L2_REGULARIZER = 1e-5
MOMENTUM = 0.99
WEIGHT_DECAY = 5e-4         # L2 regularization
LOSS_TYPE = 'CrossEntropy'
LOSS_WEIGHT = 0.5

# net architecure
LEVEL = 5
NCHANNELS = 16
DROPOUT = False
DROPOUT_RATE = 0.2
CONCATENATE = False
AM_POS = (3,)
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

CLASSES = ['anterior', 'none', 'posterior']
CLASS_COLUMN = 'position'
N_CLASSES = len(CLASSES)

SEED = 42
torch.manual_seed(SEED)

model = Net(inchannels=1, n_classes=N_CLASSES, first_channels=NCHANNELS,
            image_size=tuple(RESAMPLE_SIZE), levels=LEVEL,
            dropout=DROPOUT, dropout_rate=DROPOUT_RATE,
            concatenation=CONCATENATE,
            attention_layers_pos=AM_POS)

# -------- Optimizer + Scheduler ----------------------------------------------------------

if torch.cuda.is_available():
    print('CUDA')
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)
print('optimizer lr = {}'.format(LEARNING_RATE))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
