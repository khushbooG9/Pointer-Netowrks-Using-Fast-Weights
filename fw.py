import tensorflow as tf 
import numpy as np 
tf.enable_eager_execution() 



class fastweights(object):
	def __init__(self, inputt, output_size, batch_size=128, decay_rate = 0.9, learning_rate = 0.5, hidden_size=512):
		super(fastweights, self).__init__()
		self.X = inputt
		self.Y_size = output_size
		self.batch_size = batch_size
		self.DR = decay_rate
		self.LR = learning_rate
		self.hidden_size = hidden_size
		self.W_x = tfe.Variable(tf.random_uniform([inputt[1], hidden_size], -np.sqrt(2/input_size), np.sqrt(2/input_size)), dtype=float32)
		self.B_x = tfe.Variable(tf.zeros(hidden_size), dtype=float32)
		self.W_h = tfe.Variable(initial_value = 0.5 * np.identity(hidden_size), dtype = float32)
		self.W_y = tfe.Variable(tf.random_uniform([hidden_size, output_size], -np.sqrt(2/hidden_size), np.sqrt(2/hidden_size)), dtype = float32)
		self.B_y = tfe.Variable(tf.zeros(output_size), dtype=float32)
		self.scale = tfe.Variable(tf.ones(hidden_size), dtype =float32)
		self.shift = tfe.Variable(tf.zeros(hidden_size), dtype = float32) 
		#initial values of A and H matricies
		self.A = tf.zeros([self.batch_size, self.hidden_size,self.hidden_size], dtype = float32)
		self.H = tf.zeros([self.batch_size, self.hidden_size], dtype = float32)
    
	
	def call(self,S=1):
    
		for t in range(self.X.shape[1]):
			#first hidden state, A and H_s are  zero at this point so the part A(t)H_s(t+1) becomes zero
			self.H = tf.nn.relu((tf.matmul(self.H,self.W_h))+(tf.matmul(self.X[:, t, :],self.W_x)+self.B_x))
			#reshaping to use it with A, to calculate the A(t)H_s(t+1)
			H_s = tf.reshape(self.H, [self.batch_size, 1, self.hidden_size])
			#Initial A for this particular time step: A(t) = decay*A(t-1)+ learning*h(t)h(t).T
			#self.A = tf.add((tf.scalar_mul(self.DR, self.A)),(tf.batch_matmul(tf.transpose(self.H_s, [0,2,1]),self.H_s)))
			self.A = (tf.scalar_mul(self.DR, self.A))+ tf.scalar_mul(self.LR,(tf.batch_matmul(tf.transpose(H_s, [0,2,1]),H_s)))
			#inner loop for fast weights, tfor S steps
			for _ in range(S):
				#calculating H_s without the non linearity first, so we can use linear normalization 
				H_s = tf.reshape(tf.matmul(self.H,self.W_h),tf.shape(H_s)) + tf.reshape(tf.matmul(self.X[:,t,:],self.W_x)+self.B_x,tf.shape(H_s)) + tf.batch_matmul(H_s, self.A)
				#Applying Layer Normalization 
				mean, var = tf.nn.moments(H_s, axes =2, keep_dims = True)
				H_s = (self.scale*(H_s - mean))/(tf.sqrt(var + 1e-5) + self.shift)
				#applying non linearity
				H_s = tf.nn.relu(H_s)
			self.H = tf.reshape(H_s,[self.batch_size, self.hidden_size])
		finallayer = tf.matmul(self.H, self.W_y) + self.B_y
		return finallayer





