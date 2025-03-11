import time

NUM_WORKER = 10
NUM_CLASSES = 5
EPOCHS = 10
BATCH_SIZE = 8
LR = 0.001
LRF = 0.01

DEVICE = 'cuda:1'

PROJECT_DIR = '/home/llb/workspace/'
RUN_TIME = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

OUTPUT_PATH = PROJECT_DIR + 'my_vit/output/' + RUN_TIME + '/'

# 日志路径
LOG_DIR = OUTPUT_PATH + "logs/"
LOG_PATH = LOG_DIR + 'log.log'

# 图像输出路径
IMAGES_PATH = OUTPUT_PATH + 'images/'

# TensorBoard 日志路径
TENSORBOARD_PATH = OUTPUT_PATH + 'tensorboard'

# 模型保存路径
WEIGHT_STORAGE_DIR = OUTPUT_PATH + 'weights/'
WEIGHT_STORAGE_PATH = WEIGHT_STORAGE_DIR + 'model-' + RUN_TIME + '.pth'

MODEL_NAME = 'vit'

FREEZE_LAYERS = True


# 数据集
DATA_DIR = "my_vit/data/"
DATASET_NAME = "flower_photos"

DATA_PATH = DATA_DIR + DATASET_NAME

# 预训练权重
PRE_WEIGHT_DIR = PROJECT_DIR + 'my_vit/pre_weights/'
PRE_WEIGHT_NAME = 'jx_vit_base_patch16_224_in21k-e5005f0a.pth'

PRE_WEIGHT_PATH = PRE_WEIGHT_DIR + PRE_WEIGHT_NAME



# 分类索引
CLASS_INDICES_PATH =  PROJECT_DIR + 'my_vit/utils/class_indices.json'


