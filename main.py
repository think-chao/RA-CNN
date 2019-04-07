from tools.gen_tfrecords import read_tfrecords
#from model.racnn import RA_CNN
from model.vgg16 import Vgg16 
from config import cfg
from model.racnn import RA_CNN
import tensorflow as tf
from PIL import Image 
import cv2
import os
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


def train():

	ra_cnn = RA_CNN()


	#net = RA_CNN()


	#with tf.Session() as sess:
	#	coord = tf.train.Coordinator()
	#	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	#	image, label = sess.run([image_batch, label_batch])

if __name__ == '__main__':
	train()
