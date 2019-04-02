import tensorflow as tf 
import numpy as np 
from layernorm import LayerNorm as ln 

class fastweights(object):
	def __init__(self, input_size, output_size, hidden_size=512):
		self.X = tf.placeholder(tf.float32, shape = [None, input_size, output_size], name = "X")
		self.Y = tf.placeholder(tf.float32, shape = [None, output_size], name = "Y")
		self.DR = tf.placeholder(tf.float32, shape = [], name = "Decay_rate")
		self.LR = tf.placeholder(tf.float32, shape = [], name = "Learning_rate")

		with tf.variable_scope("fastweights"):
			self.W_x = tf.variable(tf.random_uniform([input_size, hidden_size], -np.sqrt(2/input_size), np.sqrt(2/input_size)), dtype=float32)
			self.B_x = tf.variable(tf.zeros(hidden_size), dtype=float32)

			self.W_h = tf.variable(initial_value = 0.5 * np.identity(hidden_size), dtype = float32)

			self.W_y = tf.variable(tf.random_uniform([hidden_size, output_size], -np.sqrt(2/hidden_size), np.sqrt(2/hidden_size)), dtype = float32)
			self.B_y = tf.get_variable(tf.zeros(output_size), dtype=float32)

			self.scale = tf.variable(tf.ones(hidden_size), dtype =float32)
			self.shift = tf.variable(tf.zeros(hidden_size), dtype = float32)

		batch_size = tf.shape(self.X)[0]

		#initial values of A and H matricies
		self.A = tf.zeros([batch_size, hidden_size, hidden_size], dtype = float32)
		self.H = tf.zeros([batch_size, hidden_size], dtype = float32)

		for t in range(input_size):
			#first hidden state, A and H_s are  zero at this point so the part A(t)H_s(t+1) becomes zero
			self.H = tf.nn.relu((tf.matmul(self.H,self.W_h))+(tf.matmul(self.X[:, t, :],self.W_x)+self.B_x))
			#reshaping to use it with A, to calculate the A(t)H_s(t+1)
			self.H_s = tf.reshape(self.H, [batch_size, 1, hidden_size])
			#Initial A for this particular time step: A(t) = decay*A(t-1)+ learning*h(t)h(t).T
			#self.A = tf.add((tf.scalar_mul(self.DR, self.A)),(tf.batch_matmul(tf.transpose(self.H_s, [0,2,1]),self.H_s)))
			self.A = (tf.scalar_mul(self.DR, self.A))+(tf.batch_matmul(tf.transpose(self.H_s, [0,2,1]),self.H_s))
			#inner loop for fast weights, t=0 to t=1, so range is 1
			for i in range(1):
				#calculating H_s without the non linearity first, so we can use linear normalization 
				self.H_s = tf.reshape(tf.matmul(self.H,self.W_h,tf.shape(self.H_s))) + tf.reshape(tf.matmul(self.X[:,t,:],self.W_x)+self.B_x,tf.shape(self.H_s)) + tf.batch_matmul(self.H_s, self.A)
				LN = ln(self.H_s)
				self.H_s = LN.ln()
				#applying non linearity
				self.H_s = tf.nn.relu(self.H_s)
			self.H = tf.reshape(self.H_s,[batch_size, hidden_size])
		self.finallayer = tf.matmul(self.H, self.W_y) + self.B_y
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.finallayer,self.Y))