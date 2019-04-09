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
			tx, ty, tl = fc3[:, 0], fc3[:, 1], fc3[:, 2]
			return tx, ty, tl

		def class_softmax(input):
			class_prob = tf.layers.dense(input, units=cfg.ARCH.NUM_CLASSES, activation=tf.nn.softmax)
			return class_prob

		input = tf.placeholder('float', shape=(cfg.ARCH.BATCH_SIZE, 224, 224, 3))
		apn_input, softmax_input = feature_extractor(input)
		apn_output = apn(apn_input)
		print('apn_output is {}'.format(apn_output))
		class_prob = class_softmax(softmax_input)
		next_stage_input = APN_CROP().forward(input, apn_output)
		return True


class APN_CROP():
	def forward(self, images, locs):
		in_size = images.shape[2]
		h = lambda x: 1 / (1+tf.exp(-10*x))
		unit = tf.stack([tf.range(0, in_size)*in_size])
		x = tf.stack([tf.transpose(unit)]*3)
		y = tf.stack([unit]*3)

		for i in range(images.shape[0]):
			print('第{}***********'.format(i))
			tx, ty, tl = locs[0][i], locs[1][i], locs[2][i]
			print(tx, ty, tl)
			os._exit(0)
			tx = tx if tx > in_size/3 else in_size/3
			tx = tx if (in_size/3*2) < tx else in_size/3*2
			ty = ty if ty > in_size/3 else (in_size / 3)
			ty = ty if (in_size/3*2) < ty else in_size/3*2
			tl = tl if tl > (in_size/3) else in_size/3

			w_off = int(tx-tl) if (tx-tl) > 0 else 0
			h_off = int(ty-tl) if (ty-tl) > 0 else 0
			w_end = int(tx+tl) if (tx+tl) < in_size else in_size
			h_end = int(ty+tl) if (ty+tl) < in_size else in_size

			mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
			xatt = images[i] * mk
			xatt_cropped = xatt[:, h_off : h_end, w_off : w_end]
		return
