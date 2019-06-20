import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd 
import scipy.sparse as scp 

#Used to manage datasets
class DataHandler :

	def __init__(self, directory_path):
		self.R = scp.load_npz(directory_path +  '/' + 'rating.npz').todense()
		self.mask = scp.load_npz(directory_path +  '/' + 'train_mask.npz').todense()


	def batch_init(self, batch_size) : #initialize dataset batch control variables
		self.batch_beg = 0
		self.batch_end = 0
		self.batch_size = 0

		self.batch_end += batch_size
		self.batch_size  = batch_size





	def next_batch(self):# produces the next batch of rows from the datasets sequentially as when its called

		batch_R = self.R[self.batch_beg:self.batch_end, :]
		batch_mask = self.mask[self.batch_beg:self.batch_end, :]
		batch_mask_inverse = np.where(batch_mask , 0 , 1)

		self.batch_beg+=self.batch_size
		self.batch_end+=self.batch_size

		return batch_R, batch_mask, batch_mask_inverse

	def get_num_batch(self) :

		return int(self.R.shape[0]/self.batch_size)

		



class DAPL :

	def __init__(self, Dataset = None , learning_rate = 0.1 , epochs = 10 , batch_size = 20, shape = (0,0)) :

		#Datasets
		self.Dataset = Dataset

		#Parameters
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.shape = shape

		#Training Paceholders
		self.X = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.X_mask = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.X_mask_inverse = tf.placeholder(tf.float32, [None, self.shape[1]])

		#Session variables
		self.sess = None
		self.init_op = tf.global_variables_initializer()


	def network_weights_biases(self,num_nodes) :

		self.weights = []
		self.biases = []

		initializer=tf.variance_scaling_initializer()

			
		for i in range(len(num_nodes)-1) :

			axis_0 = num_nodes[i]
			axis_1 = num_nodes[i+1]

			#print('-----------------------------------------------------------')
			#print('axis_0: ', axis_0,"  axis_1: ", axis_1)
			#print('-----------------------------------------------------------')

			W = tf.Variable(initializer([axis_0,axis_1]), name = 'W' + str(i))
			b = tf.Variable(tf.zeros([axis_1]), name = 'b' + str(i))

			self.weights.append(W)
			self.biases.append(b)





	def network_func(self) :

		input_tensor = self.X

		for i in self.biases :
			print("weights: ", i)

		#print('\n\n')

		for i in range(len(self.weights)) :

			layer = tf.add(tf.matmul(input_tensor,self.weights[i]),self.biases[i])
			layer = tf.nn.relu(layer)

			#print('layer: ', layer)

			input_tensor = layer

		#reconstructed matrix
		output = input_tensor

		return output

	def loss_func(self,X,recons_X) :

		#self.loss=tf.sqrt(tf.reduce_mean(tf.square(recons_X-self.X)))
		#self.loss=tf.reduce_mean(tf.square(recons_X-self.X))
		self.loss=tf.reduce_mean(tf.square(self.X_mask_inverse*(recons_X-self.X)))

	def optimizer_func(self) :

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

	def train(self) :

		recons_X = self.network_func()
		self.loss_func(self.X, recons_X)
		self.optimizer_func()

		init_op = tf.global_variables_initializer()

		with tf.Session() as sess :

			sess.run(init_op)

			for epoch in range(self.epochs) :

				l = 0
				self.Dataset.batch_init(batch_size = self.batch_size)
				total_batch = self.Dataset.get_num_batch()
				for i in range(total_batch) :

					batch_x, batch_mask, batch_mask_inverse = self.Dataset.next_batch()

					#print("X: ",type(batch_x)," ",batch_x.shape,"\n\n")
					#print("y: ",type(batch_y)," ",batch_y.shape,"\n\n\n")


					_, l = sess.run([self.optimizer, self.loss], feed_dict = {self.X : batch_x, self.X_mask : batch_mask, self.X_mask_inverse : batch_mask_inverse})

				 

				print("Epoch: ", epoch + 1, "cost: ", "{:.3f}".format(l))


def main() :

	#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	Dataset = DataHandler('./Experiment/BioDataset1')
	obj = DAPL(Dataset = Dataset, learning_rate = 0.01 , epochs = 10 , batch_size = 150, shape = (None,1200))

	obj.network_weights_biases([1200,600,300,600,1200])
	obj.train()

main()







