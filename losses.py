# -*- coding: utf-8 -*-

import tensorflow as tf

def pw_softmaxwithloss_2d(y_true, y_pred):
	
	exp_pred = tf.exp(y_pred)
	
	try:
		
		sum_exp = tf.reduce_sum(exp_pred, 3, keepdims=True)
	
	except:
		
		sum_exp = tf.reduce_sum(exp_pred, 3, keep_dims=True)
	
	tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(y_pred)[3]]))
	softmax_output = tf.div(exp_pred, tensor_sum_exp)
	ce = -tf.reduce_mean(y_true * tf.log(tf.clip_by_value(softmax_output, 1e-12, 1.0)))
	
	return softmax_output, ce

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
	
	try:
		
		pk = tf.reduce_sum(y_true * y_pred, 3, keepdims=True)
	
	except:
		
		pk = tf.reduce_sum(y_true * y_pred, 3, keep_dims=True)
	
	fl = -alpha * tf.reduce_mean(tf.pow(1.0 - pk, gamma) * tf.log(pk))
	
	return fl
