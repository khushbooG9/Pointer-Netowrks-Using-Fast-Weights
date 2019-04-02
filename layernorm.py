import numpy as np 
import tensorflow as tf 

#Layer normalization using mean and variance to recenter and rescale the inputs
class LayerNorm(object):
	def __init__(self,inputs, epsilon =1e-5, scope=None ):
		self.inputs = inputs
		self.epsilon = epsilon
		self.scope = scope

	def meanvar(self):
		mean, var = tf.nn.moments(self.inputs, [1], keep_dims = True)
		return mean, var

	def ln(self):
		mean , var = self.meanvar(self.inputs)
		with tf.variable_scope(self.scope + "LN"):
			scale = tf.get_variable('alpha', shape=[self.inputs.get_shape()[1]], initializer=tf.constant_initializer(1), dtype = float32)
			shift = tf.get_variable('beta',shape=[self.inputs.get_shape()[1]], initializer=tf.constant_initializer(0), dtype=float32)
		LN = scale * (self.inputs - mean) / tf.sqrt(var + self.epsilon) + shift
		return LN 
