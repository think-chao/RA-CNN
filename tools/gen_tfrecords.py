#coding=utf-8
import tensorflow as tf
import os.path, sys
import shutil
import pandas as pd
import cv2
from tqdm import tqdm

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')
from config import cfg 

# data to int64List
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
# data to floatlist
def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))
# data to byteslist
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def organize_data_folder(label_path, data_root):
	df = pd.read_csv(label_path)
	all_breeds = set(df['breed'].values)
	for breed in tqdm(all_breeds):
		os.makedirs(os.path.join(data_root, breed))
		imgs = df[df['breed'] == breed]['id'].values
		for im in imgs:
			shutil.move(os.path.join(data_root, im+'.jpg'), os.path.join(data_root, breed))

def get_num_classes(labels_path):
	df = pd.read_csv(labels_path)
	return set(df['breed'].values)


def gen_tfrecords(data_root, tf_save_path, labels_path):
	write = tf.python_io.TFRecordWriter(tf_save_path)
	for index, name in tqdm(enumerate(os.listdir(data_root))):
		sub_class_dir = os.path.join(data_root, name)
		for im_name in os.listdir(sub_class_dir):
			im = cv2.imread(os.path.join(sub_class_dir, im_name))
			im = cv2.resize(im, (224, 224))
			im_raw = im.tobytes()
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': _int64_feature(index),
				'im_raw': _bytes_feature(im_raw)
				}))
			write.write(example.SerializeToString())
	write.close()


def read_tfrecords(filename):
	filename_deque = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, SerializedExample = reader.read(filename_deque)
	features = tf.parse_single_example(SerializedExample, features={
		'label': tf.FixedLenFeature([], tf.int64),
		'im_raw': tf.FixedLenFeature([], tf.string)
		})
	img = tf.decode_raw(features['im_raw'], tf.uint8)
	img = tf.reshape(img, [224, 224, 3])
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)
	return img, label

if __name__ == '__main__':

	#organize_data_folder(cfg.PATH.LABELS, cfg.PATH.DATA)
	gen_tfrecords(cfg.PATH.DATA, cfg.PATH.TF_RECORDS_SAVE, cfg.PATH.LABELS)
	#img, label = read_tfrecords(cfg.PATH.TF_RECORDS_SAVE)
	#print(img)
