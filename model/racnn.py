#coding=utf-8
import tensorflow as tf
import os, sys

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')
from config import cfg
from model.vgg16 import Vgg16

class RA_CNN(object):
	"""docstring for RA_CNN"""
	def __init__(self):
		super(RA_CNN, self).__init__()
		self.__session = tf.Session()

		# Building graph
		with self.__session.as_default():
		 	flag = self.crnn()
		 	print(flag)


	def crnn(self):
		def feature_extractor(input):
			vgg16 = Vgg16()
			conv5_3, relu7 = vgg16.build(input)
			return conv5_3, relu7

		def apn(input):
			conv = tf.layers.conv2d(input, filters=64, kernel_size=(3, 3), padding = "same", activation=tf.nn.relu)
			fc1 = tf.layers.flatten(conv)
			fc2 = tf.layers.dense(fc1, units=512, activation=tf.nn.relu)
			fc3 = tf.layers.dense(fc2, units=3, activation=tf.nn.softmax)
			print(fc1, fc2, fc3)
			tx, ty, tl = fc3[:, 0], fc3[:, 1], fc3[:, 2],
			return tx, ty, tl

		def apn_crop(original, loc):
			pass

		input = tf.placeholder('float', shape=(cfg.ARCH.BATCH_SIZE, 224, 224, 3))
		feature_stage_one, class_stage_one = feature_extractor(input)
		apn_output = apn(feature_stage_one)
		next_stage_input = apn_crop(apn_output, apn_output)
		return True


