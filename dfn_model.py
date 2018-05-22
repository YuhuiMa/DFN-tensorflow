# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from subnetworks import *
from losses import *

def cal_iou(y_true, y_pred):
	
	overlap_map = y_true * y_pred
	
	try:
		
		gt_count = tf.reduce_sum(tf.cast(y_true, tf.float32), [1, 2], keepdims=True)
		pred_count = tf.reduce_sum(tf.cast(y_pred, tf.float32), [1, 2], keepdims=True)
		overlap_count = tf.reduce_sum(tf.cast(overlap_map, tf.float32), [1, 2], keepdims=True)
	
	except:
		
		gt_count = tf.reduce_sum(tf.cast(y_true, tf.float32), [1, 2], keep_dims=True)
		pred_count = tf.reduce_sum(tf.cast(y_pred, tf.float32), [1, 2], keep_dims=True)
		overlap_count = tf.reduce_sum(tf.cast(overlap_map, tf.float32), [1, 2], keep_dims=True)
	
	iou = tf.div(overlap_count, gt_count + pred_count - overlap_count)
	
	return iou

class DFN(object):
	
	def __init__(self, max_iter, batch_size=32, init_lr=0.004, power=0.9, momentum=0.9, stddev=0.02, regularization_scale=0.0001, alpha=0.25, gamma=2.0, fl_weight=0.1):
		
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.init_lr = init_lr
		self.power = power
		self.momentum = momentum
		self.stddev = stddev
		self.regularization_scale = regularization_scale
		self.alpha = alpha
		self.gamma = gamma
		self.fl_weight = fl_weight
		self.graph = tf.Graph()
		with self.graph.as_default():
			
			self.X = tf.placeholder(tf.float32, shape=(self.batch_size, 512, 512, 3))
			self.Y = tf.placeholder(tf.float32, shape=(self.batch_size, 512, 512, NUM_CLASSES))

			self.build_arch()
			self.loss()
			self.evaluation()
			self._summary()
			
			self.global_iter = tf.Variable(0, name='global_iter', trainable=False)
			self.lr = tf.train.polynomial_decay(self.init_lr, self.global_iter, self.max_iter, end_learning_rate=0.0, power=self.power)
			self.optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum)
			self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_iter)
			self.saver = tf.train.Saver()
		
		tf.logging.info('Setting up the main structure')
	
	def build_arch(self):
		
		######### -*- ResNet-101 -*- #########
		with tf.variable_scope("resnet_101"):
			
			self.ib_2, self.ib_3, self.ib_4, self.ib_5, self.global_avg_pool = nn_base(self.X, k=0, initializer=tf.random_normal_initializer(0, self.stddev), regularizer=tf.contrib.layers.l2_regularizer(self.regularization_scale))
		
		######### -*- Smooth Network -*- #########
		with tf.variable_scope("smooth"):
			
			self.b1, self.b2, self.b3, self.b4, self.fuse = nn_smooth(self.ib_2, self.ib_3, self.ib_4, self.ib_5, self.global_avg_pool, k=0, initializer=tf.random_normal_initializer(0, self.stddev), regularizer=tf.contrib.layers.l2_regularizer(self.regularization_scale))
		
		######### -*- Border Network -*- #########
		with tf.variable_scope("border"):
			
			self.o = nn_border(self.ib_2, self.ib_3, self.ib_4, self.ib_5, k=0, initializer=tf.random_normal_initializer(0, self.stddev), regularizer=tf.contrib.layers.l2_regularizer(self.regularization_scale))
	
	def loss(self):
		
		######### -*- Softmax Loss -*- #########
		self.softmax_b1, self.ce1 = pw_softmaxwithloss_2d(self.Y, self.b1)
		self.softmax_b2, self.ce2 = pw_softmaxwithloss_2d(self.Y, self.b2)
		self.softmax_b3, self.ce3 = pw_softmaxwithloss_2d(self.Y, self.b3)
		self.softmax_b4, self.ce4 = pw_softmaxwithloss_2d(self.Y, self.b4)
		self.softmax_fuse, self.cefuse = pw_softmaxwithloss_2d(self.Y, self.fuse)
		self.total_ce = self.ce1 + self.ce2 + self.ce3 + self.ce4 + self.cefuse
		
		######### -*- Focal Loss -*- #########
		self.fl = focal_loss(self.Y, self.o, alpha=self.alpha, gamma=self.gamma)
		
		######### -*- Total Loss -*- #########
		self.total_loss = self.total_ce + self.fl_weight * self.fl
	
	def evaluation(self):
		
		self.prediction = tf.argmax(self.fuse, axis = 3)
		self.ground_truth = tf.argmax(self.Y, axis = 3)
		self.iou = cal_iou(self.ground_truth, self.prediction)
		self.mean_iou = tf.reduce_mean(self.iou)
	
	def _summary(self):
		
		trainval_summary = []
		trainval_summary.append(tf.summary.scalar('softmax_loss', self.total_ce))
		trainval_summary.append(tf.summary.scalar('focal_loss', self.fl))
		trainval_summary.append(tf.summary.scalar('total_loss', self.total_loss))
		trainval_summary.append(tf.summary.scalar('mean_iou', self.mean_iou))
		self.trainval_summary = tf.summary.merge(trainval_summary)
