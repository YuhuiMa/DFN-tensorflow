# -*- coding: utf-8 -*-

import tensorflow as tf

def batchnorm(inputs):
	
	return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def lrelu(x, a):
	
	with tf.name_scope("lrelu"):
		# adding these together creates the leak part and linear part
		# then cancels them out by subtracting/adding an absolute value term
		# leak: a*x/2 - a*abs(x)/2
		# linear: x/2 + abs(x)/2
		
		# this block looks like it has 2 inputs on the graph unless we do this
		x = tf.identity(x)
		return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def side_branch(x, nc, factor, initializer=tf.random_normal_initializer(0, 0.02), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), bias_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
	
	y = tf.layers.conv2d(x, nc, kernel_size=1, strides=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	return tf.layers.conv2d_transpose(y, nc, kernel_size=2*factor, strides=(factor, factor), padding="same", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

def identity_block(batch_input, filters, kernel_size=3, k=0, initializer=tf.random_normal_initializer(0, 0.02), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), bias_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
	
	nb_filter1, nb_filter2, nb_filter3 = filters
	
	res_branch2a = tf.layers.conv2d(batch_input, nb_filter1, kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	bn_branch2a = batchnorm(res_branch2a)
	lrelu_branch2a = lrelu(bn_branch2a, k)
	
	res_branch2b = tf.layers.conv2d(lrelu_branch2a, nb_filter2, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	bn_branch2b = batchnorm(res_branch2b)
	lrelu_branch2b = lrelu(bn_branch2b, k)
	
	res_branch2c = tf.layers.conv2d(lrelu_branch2b, nb_filter3, kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	bn_branch2c = batchnorm(res_branch2c)
	
	return lrelu(batch_input + bn_branch2c, k)

def conv_block(batch_input, filters, kernel_size=3, strides=(2, 2), k=0, initializer=tf.random_normal_initializer(0, 0.02), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), bias_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
	
	nb_filter1, nb_filter2, nb_filter3 = filters
	
	######### -*- branch 1 -*- #########
	res_branch1 = tf.layers.conv2d(batch_input, nb_filter3, kernel_size=1, strides=strides, padding="valid", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	bn_branch1 = batchnorm(res_branch1)
	
	######### -*- branch 2 -*- #########
	res_branch2a = tf.layers.conv2d(batch_input, nb_filter1, kernel_size=1, strides=strides, padding="valid", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	bn_branch2a = batchnorm(res_branch2a)
	lrelu_branch2a = lrelu(bn_branch2a, k)
	
	res_branch2b = tf.layers.conv2d(lrelu_branch2a, nb_filter2, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	bn_branch2b = batchnorm(res_branch2b)
	lrelu_branch2b = lrelu(bn_branch2b, k)
	
	res_branch2c = tf.layers.conv2d(lrelu_branch2b, nb_filter3, kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	bn_branch2c = batchnorm(res_branch2c)
	
	return lrelu(bn_branch1 + bn_branch2c, k)

######### -*- Refinement Residual Block -*- #########
def rrb(batch_input, filters, kernel_size=3, k=0, initializer=tf.random_normal_initializer(0, 0.02), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), bias_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
	
	nb_filter1, nb_filter2 = filters
	
	refine_input = tf.layers.conv2d(batch_input, nb_filter2, kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	res_branch2a = tf.layers.conv2d(refine_input, nb_filter1, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	bn_branch2a = batchnorm(res_branch2a)
	lrelu_branch2a = lrelu(bn_branch2a, k)
	res_branch2b = tf.layers.conv2d(lrelu_branch2a, nb_filter2, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	
	return lrelu(refine_input + res_branch2b, k)

######### -*- Channel Attention Block -*- #########
def cab(batch_input1, batch_input2, fn, k=0, initializer=tf.random_normal_initializer(0, 0.02), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), bias_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
	
	assert batch_input1.get_shape().as_list() == batch_input2.get_shape().as_list()
	
	batch_input = tf.concat([batch_input1, batch_input2], axis=3)
	global_avg_pool = tf.nn.avg_pool(batch_input, ksize=[1, batch_input.get_shape().as_list()[1], batch_input.get_shape().as_list()[2], 1], strides=[1, batch_input.get_shape().as_list()[1], batch_input.get_shape().as_list()[2], 1], padding='SAME')
	conv_1 = tf.layers.conv2d(global_avg_pool, fn, kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	activation_1 = lrelu(conv_1, k)
	conv_2 = tf.layers.conv2d(activation_1, batch_input1.get_shape().as_list()[-1], kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
	activation_2 = tf.sigmoid(conv_2)
	mul = tf.multiply(batch_input1, activation_2)
	
	return batch_input2 + mul
