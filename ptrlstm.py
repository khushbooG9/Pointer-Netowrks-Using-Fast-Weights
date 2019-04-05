import tensorflow as tf 
#from tensorflow.keras.layers import LSTM
import tensorflow.contrib.eager as tfe
import numpy as np 
tf.enable_eager_execution()

class Encoder(tf.keras.Model):
	def __init__(self, hidden_size=512):
		super(Encoder, self).__init__()
		self.encoder = LSTM(hidden_size, return_sequence = True, return_state = True)

	def call(self, x):
		e, state_h, state_c  = self.encoder(x)        
		return e, [state_h, state_c]

class Decoder(tf.keras.Model):
	def __init__(self, hidden_size=512):
		super(Decoder, self).__init__()
		self.decoder = LSTM(hidden_size, return_sequence = True, return_state = True)

	def call(self, x, hidden_states):
		d, state_h, state_c  = self.decoder(x, initial_state=hidden_states)
		return d, [state_h, state_c]

class PtrnetLSTM(tf.keras.Model):
	def __init__(self, hidden_size=512):
		super(PtrnetLSTM, self).__init__()
		#self.W1 = tfe.variable(tf.random_uniform([hidden_size, hidden_size], -0.08, 0.08), dtype=float32)
		self.W1 = tf.keras.layers.Dense(hidden_size, kernel_initializer= tf.keras.initializers.RandomUniform(minval = -0.08, maxval = 0.08, seed = None), use_bias=False)
		self.W2 = tf.keras.layers.Dense(hidden_size, kernel_initializer= tf.keras.initializers.RandomUniform(minval = -0.08, maxval = 0.08, seed = None), use_bias=False)
		#Dense layer -> dot(input, kernel) -> so now Ui = vT . tanh(W1 . e + W2 . di)  becomes Ui = tanh(e . W1 + di . W2) . v
		self.VT = tf.keras.layers.Dense(1, use_bias=False)
	# e= encoder output and d = decoder output
	def call(self, e,d):
		u = self.VT(tf.nn.tanh(self.W1(e) + self.W2(d)))
		attention = tf.nn.softmax(u, axis = 1)

		return tf.reshape(attention, attention.shape[0], attention.shape[1])


