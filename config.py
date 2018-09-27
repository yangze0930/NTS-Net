BATCH_SIZE = 4
PROPOSAL_NUM = 6
CAT_NUM = 4
INPUT_SIZE = (448, 448)  # (w, h)
LR = 0.001
WD = 1e-4
SAVE_FREQ = 1
MEAN = [-0.0886, -0.0369,  0.0750]
STD = [0.5113, 0.5325, 0.5322]
resume = ''
test_model = 'model.ckpt'
# save_dir = '/data/mltest/train-repositorys/NTS-Net/model'
save_dir = 'model'
# DATASET_PATH = '/data/mltest/train-repositorys/NTS-Net/data/CUB_200_2011' # CUB dataset path is under project's path
DATASET_PATH = 'E:/car-classify-dataset/small_dataset'
dataloader_num_workers = 4
# fc nums correspond to classes
FC_NUMS = 2000