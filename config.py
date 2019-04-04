from easydict import EasyDict as edict 

cfg = edict()

cfg.PATH = edict()
cfg.PATH.DATA = r'E:\file\data\dog_breed\train'
cfg.PATH.LABELS = r'E:\file\data\dog_breed\labels.csv'
cfg.PATH.TF_RECORDS_SAVE = r'C:\Users\v_wangchao3\Desktop\code\RA-CNN\data\tfrecords\train.tfrecords'
cfg.PATH.vgg16_npy_path = r'E:\file\data\dog_breed\vgg16\vgg16.npy'


cfg.ARCH = edict()
cfg.ARCH.BATCH_SIZE = 1
cfg.ARCH.NUM_CLASSES = 120