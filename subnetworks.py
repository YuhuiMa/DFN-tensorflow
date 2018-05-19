# -*- coding: utf-8 -*-

import tensorflow as tf
from components import *

NUM_CLASSES = 2

######### -*- ResNet-101 -*- #########
def nn_base(input_tensor, k=0, initializer=tf.random_normal_initializer(0, 0.02), regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
	'''
	ResNet-101:
		conv1: 7 × 7, 64, stride 2
		conv2_x: 
			3 × 3 max pooling, stride 2
			 ————————————— 
			|  1 × 1, 64  |
			|  3 × 3, 64  | × 3
			|  1 × 1, 256 |
			 ————————————— 
		conv3_x:
			 —————————————— 
			|  1 × 1, 128  |
			|  3 × 3, 128  | × 4
			|  1 × 1, 512  |
			 —————————————— 
		conv4_x:
			 —————————————— 
			|  1 × 1, 256  |
			|  3 × 3, 256  | × 23
			|  1 × 1, 1024 |
			 —————————————— 
		conv5_x:
			 —————————————— 
			|  1 × 1, 512  |
			|  3 × 3, 512  | × 3
			|  1 × 1, 2048 |
			 —————————————— 
		pool: global average pooling
	'''
	
	batch_input = tf.layers.conv2d_transpose(input_tensor, input_tensor.get_shape().as_list()[-1], kernel_size=7, strides=(1, 1), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- conv1 -*- #########
	conv_1 = tf.layers.conv2d(batch_input, 64, kernel_size=7, strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	bn_1 = batchnorm(conv_1)
	lrelu_1 = lrelu(bn_1, k)
	
	######### -*- conv2_x -*- #########
	max_pool_1 = tf.nn.max_pool(lrelu_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	cb_2 = conv_block(max_pool_1, [64, 64, 256], kernel_size=3, strides=(1, 1), k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_2a = identity_block(cb_2, [64, 64, 256], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_2b = identity_block(ib_2a, [64, 64, 256], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- conv3_x -*- #########
	cb_3 = conv_block(ib_2b, [128, 128, 512], kernel_size=3, strides=(2, 2), k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_3a = identity_block(cb_3, [128, 128, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_3b = identity_block(ib_3a, [128, 128, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_3c = identity_block(ib_3b, [128, 128, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- conv4_x -*- #########
	cb_4 = conv_block(ib_3c, [256, 256, 1024], kernel_size=3, strides=(2, 2), k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4a = identity_block(cb_4, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4b = identity_block(ib_4a, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4c = identity_block(ib_4b, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4d = identity_block(ib_4c, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4e = identity_block(ib_4d, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4f = identity_block(ib_4e, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4g = identity_block(ib_4f, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4h = identity_block(ib_4g, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4i = identity_block(ib_4h, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4j = identity_block(ib_4i, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4k = identity_block(ib_4j, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4l = identity_block(ib_4k, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4m = identity_block(ib_4l, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4n = identity_block(ib_4m, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4o = identity_block(ib_4n, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4p = identity_block(ib_4o, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4q = identity_block(ib_4p, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4r = identity_block(ib_4q, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4s = identity_block(ib_4r, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4t = identity_block(ib_4s, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4u = identity_block(ib_4t, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_4v = identity_block(ib_4u, [256, 256, 1024], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- conv5_x -*- #########
	cb_5 = conv_block(ib_4v, [512, 512, 2048], kernel_size=3, strides=(2, 2), k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_5a = identity_block(cb_5, [512, 512, 2048], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ib_5b = identity_block(ib_5a, [512, 512, 2048], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- pool -*- #########
	global_avg_pool = tf.nn.avg_pool(ib_5b, ksize=[1, ib_5b.get_shape().as_list()[1], ib_5b.get_shape().as_list()[2], 1], strides=[1, ib_5b.get_shape().as_list()[1], ib_5b.get_shape().as_list()[2], 1], padding='SAME')
	
	return ib_2b, ib_3c, ib_4v, ib_5b, global_avg_pool

######### -*- Smooth Network -*- #########
def nn_smooth(ib_2, ib_3, ib_4, ib_5, global_avg_pool, k=0, initializer=tf.random_normal_initializer(0, 0.02), regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
	
	######### -*- stage 5 -*- #########
	cab5_input1 = rrb(ib_5, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	cab5_input2 = tf.layers.conv2d_transpose(global_avg_pool, 512, kernel_size=1, strides=(cab5_input1.get_shape().as_list()[1], cab5_input1.get_shape().as_list()[2]), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	cab5_output = cab(cab5_input1, cab5_input2, 128, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	rrb5_output = rrb(cab5_output, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- stage 4 -*- #########
	cab4_input1 = rrb(ib_4, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	cab4_input2 = tf.layers.conv2d_transpose(rrb5_output, 512, kernel_size=1, strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	b4 = side_branch(cab4_input2, NUM_CLASSES, 16, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	o4 = tf.sigmoid(b4)
	cab4_output = cab(cab4_input1, cab4_input2, 128, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	rrb4_output = rrb(cab4_output, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- stage 3 -*- #########
	cab3_input1 = rrb(ib_3, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	cab3_input2 = tf.layers.conv2d_transpose(rrb4_output, 512, kernel_size=1, strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	b3 = side_branch(cab3_input2, NUM_CLASSES, 8, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	o3 = tf.sigmoid(b3)
	cab3_output = cab(cab3_input1, cab3_input2, 128, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	rrb3_output = rrb(cab3_output, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- stage 2 -*- #########
	cab2_input1 = rrb(ib_2, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	cab2_input2 = tf.layers.conv2d_transpose(rrb3_output, 512, kernel_size=1, strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	b2 = side_branch(cab2_input2, NUM_CLASSES, 4, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	o2 = tf.sigmoid(b2)
	cab2_output = cab(cab2_input1, cab2_input2, 128, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	rrb2_output = rrb(cab2_output, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- stage 1 -*- #########
	output = tf.layers.conv2d_transpose(rrb2_output, 512, kernel_size=1, strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	b1 = side_branch(output, NUM_CLASSES, 2, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	o1 = tf.sigmoid(b1)
	
	b = tf.concat([b1, b2, b3, b4], axis=3)
	fuse = tf.layers.conv2d(b, NUM_CLASSES, kernel_size=1, strides=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	ofuse = tf.sigmoid(fuse)
	
	return o1, o2, o3, o4, ofuse

######### -*- Border Network -*- #########
def nn_border(ib_2, ib_3, ib_4, ib_5, k=0, initializer=tf.random_normal_initializer(0, 0.02), regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
	
	######### -*- stage 1 -*- #########
	input_1a = rrb(ib_2, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	input_1b = rrb(ib_3, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	input_1b = tf.layers.conv2d_transpose(input_1b, 512, kernel_size=1, strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	output1 = rrb(input_1a + input_1b, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- stage 2 -*- #########
	input2 = rrb(ib_4, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	input2 = tf.layers.conv2d_transpose(input2, 512, kernel_size=1, strides=(4, 4), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	output2 = rrb(output1 + input2, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	######### -*- stage 3 -*- #########
	input3 = rrb(ib_5, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	input3 = tf.layers.conv2d_transpose(input3, 512, kernel_size=1, strides=(8, 8), padding="valid", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	output3 = rrb(output2 + input3, [512, 512], kernel_size=3, k=k, initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	output = tf.layers.conv2d_transpose(output3, 64, kernel_size=3, strides=(2, 2), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	
	return tf.layers.conv2d_transpose(output, NUM_CLASSES, kernel_size=7, strides=(2, 2), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
