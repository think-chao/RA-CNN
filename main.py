from tools.gen_tfrecords import read_tfrecords
from model.racnn import RA_CNN
from model.vgg16 import Vgg16 
from config import cfg
import tensorflow as tf
from PIL import Image 
import cv2
import os
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


def train():

	image, label = read_tfrecords(cfg.PATH.TF_RECORDS_SAVE)
	image_batch, label_batch = tf.train.shuffle_batch([image, label], num_threads=16, batch_size=cfg.ARCH.BATCH_SIZE,
		capacity=2000, min_after_dequeue=1900)
	input = tf.placeholder('float', shape=(cfg.ARCH.BATCH_SIZE, 224, 224, 3))
	with tf.Session() as sess:
		coord=tf.train.Coordinator()
		threads= tf.train.start_queue_runners(coord=coord)
		vgg16 = Vgg16()

		with tf.name_scope("content_vgg"):
			vgg16.build(input)
		image_batch, label_batch = sess.run([image_batch, label_batch])
		prob = sess.run(vgg16.prob, feed_dict={
			input: image_batch
			})
		print(prob)


	#net = RA_CNN()


	#with tf.Session() as sess:
	#	coord = tf.train.Coordinator()
	#	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	#	image, label = sess.run([image_batch, label_batch])

if __name__ == '__main__':
	train()
