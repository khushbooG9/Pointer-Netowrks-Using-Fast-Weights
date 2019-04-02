import tensorflow as tf 
from tensorflow.keras.layers import LSTM
import numpy as np 

class PtrnetLSTM(LSTM):
	def __init__(self, input_size, output_size, hidden_size=512):
		super(PtrnetLSTM, self).__init__()
		self.encoder = LSTM(hidden_size, return_sequence = True, return_state = True)
		self.decoder = LSTM(hidden_size, return_sequence = True, return_state = True)

		with tf.variable_scope("PtrnetLSTM"):
			self.W1 = tf.variable(tf.random_uniform([hidden_size, hidden_size], -0.08, 0.08), dtype=float32)
			self.W2 = tf.variable(tf.random_uniform([hidden_size, hidden_size], -0.08, 0.08), dtype=float32)
			self.VT = tf.variable(, dtype = float32)


	def forward(self):
		
		for t in range(input_size):
			
