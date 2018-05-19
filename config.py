# -*- coding: utf-8 -*-

import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For focal loss
flags.DEFINE_float('alpha', 0.25, 'coefficient for focal loss')
flags.DEFINE_float('gamma', 2.0, 'factor for focal loss')
flags.DEFINE_float('fl_weight', 0.1, 'regularization coefficient for focal loss')

# For training
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epoch', 50, 'epoch')

flags.DEFINE_float('init_lr', 0.004, 'initial learning rate')
flags.DEFINE_float('power', 0.9, 'decay factor of learning rate')
flags.DEFINE_float('momentum', 0.9, 'momentum factor')
flags.DEFINE_float('stddev', 0.02, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.0001, 'regularization coefficient for W and b')


############################
#   environment setting    #
############################
flags.DEFINE_string('images', 'data', 'The root directory of dataset')
flags.DEFINE_boolean('is_training', True, 'train or test phase')
flags.DEFINE_string('logdir', 'logs', 'logs directory')
flags.DEFINE_string('log', 'trainval.log', 'log file')
flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(step)')
flags.DEFINE_string('models', 'models', 'path for saving models')
flags.DEFINE_string('test_outputs', 'test-outputs', 'path for saving test results')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
