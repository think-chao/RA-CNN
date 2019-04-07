from easydict import EasyDict as edict 

cfg = edict()

cfg.PATH = edict()
cfg.PATH.DATA = r'F:\data\dog_breed\image'
cfg.PATH.LABELS = r'F:\data\dog_breed\labels.csv'
cfg.PATH.TF_RECORDS_SAVE = r'C:\Users\king\Desktop\code\RA-CNN\tfrecords\train.tfrecords'
cfg.PATH.vgg16_npy_path = r'F:\data\vgg16.npy'


cfg.ARCH = edict()
cfg.ARCH.BATCH_SIZE = 2
cfg.ARCH.NUM_CLASSES = 120