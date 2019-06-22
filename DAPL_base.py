import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd 
import scipy.sparse as scp 

#Used to manage datasets
class DataHandler :

	def __init__(self, directory_path, mask_given  = False):
		self.R = scp.load_npz(directory_path +  '/' + 'rating.npz').todense()
		self.mask_given = mask_given
		if self.mask_given :
			self.mask = scp.load_npz(directory_path +  '/' + 'train_mask.npz').todense()
		else :
			self.mask = None

	#-----------------------------------------BATCH CONTROL----------------------------

	def batch_init(self, batch_size) : #initialize dataset batch control variables
		self.batch_beg = 0
		self.batch_end = 0
		self.batch_size = 0

		self.batch_end += self.R.shape[0]%batch_size
		self.batch_size  = batch_size

		num_batches = self.get_num_batch()

		return num_batches



	def next_batch(self):# produces the next batch of rows from the datasets sequentially as when its called

		batch_R = self.R[self.batch_beg:self.batch_end, :]


		#print("Batch R:", batch_R.shape)

		return batch_R

	def next_batch_mask(self, row_size = 0, missing_perc = 0.1) :#produce batch for mask matrix or produces a random generated batch using missing_perc

		if self.mask_given :
			batch_mask = self.mask[self.batch_beg:self.batch_end, :]
			#print('batch_mask.shape: ',batch_mask.shape)

		else :
			#print('row_size', row_size)
			batch_mask = np.random.binomial(1, missing_perc, size=row_size*self.R.shape[1]).reshape(row_size, self.R.shape[1])

		batch_mask_inverse = np.where(batch_mask , 0 , 1)

		return batch_mask,batch_mask_inverse

	def inc_batch(self) : #used for iterating through the datatset batchwise

		#print('self.batch_beg', self.batch_beg)
		#print('self.batch_end', self.batch_end)

		self.batch_beg =self.batch_end
		self.batch_end+=self.batch_size

	def get_num_batch(self) :#returns no of batches in the dataset

		num_batches = self.R.shape[0]//self.batch_size

		num_final_batch = self.R.shape[0] - num_batches*self.batch_size #number of instances in the final batch if the row size is not a perfect multiple of batch_size

		if num_final_batch != 0 :
			num_batches+=1

		return num_batches


	def compile_batches(self,batch, full_mat): #combines batches to produce the full matrix

		full_mat = np.append(full_mat,batch, axis = 0)

		return full_mat
		

	#--------------------------------------------------------------------------------

	#--------------------------------SAVE DATA----------------------------------------

	def save_matrix(self,file_path, matrix) :#save the matrix as .npy file in the file_path
		np.save(file_path,matrix)

	def create_dir(self,dir_path) : 
		if not os.path.exists(dir_path) :
			os.mkdir(dir_path)

	#----------------------------------------------------------------------------------



		



class DAPL :

	def __init__(self, Dataset = None , learning_rate = 0.1 , epochs = 10 , batch_size = 20, shape = (0,0), missing_perc = 0.1) :

		#Datasets
		self.Dataset = Dataset

		#Parameters
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.shape = shape
		self.missing_perc = missing_perc

		#Training Paceholders
		self.X = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.X_mask = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.X_mask_inverse = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.input_X = tf.placeholder(tf.float32, [None, self.shape[1]])

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

		input_tensor = self.input_X
		#input_tensor = self.X

		for i in self.biases :
			print("weights: ", i)

		#print('\n\n')

		for i in range(len(self.weights)) :

			layer = tf.add(tf.matmul(input_tensor,self.weights[i]),self.biases[i])
			layer = tf.nn.relu(layer)

			#print('layer: ', layer)

			input_tensor = layer

		#reconstructed matrix
		self.output = input_tensor

		return self.output

	def loss_func(self,X,recons_X) :

		#self.loss=tf.sqrt(tf.reduce_mean(tf.square(recons_X-self.X)))
		#self.loss=tf.reduce_mean(tf.square(recons_X-self.X))
		self.loss=tf.sqrt(tf.reduce_mean(tf.square(self.X_mask_inverse*(recons_X-self.X))))

	def optimizer_func(self, optimizer) :

		self.optimizer = optimizer(self.learning_rate).minimize(self.loss)

	def train(self, save_results = False, results_filePath = './results') :

		recons_X = self.network_func()
		self.loss_func(self.X, recons_X)
		self.optimizer_func(tf.train.AdamOptimizer)

		full_recons_matrix = np.empty(shape = (0,self.shape[1]))

		init_op = tf.global_variables_initializer()

		with tf.Session() as sess :

			sess.run(init_op)

			for epoch in range(self.epochs) :

				l = 0
				total_batch = self.Dataset.batch_init(batch_size = self.batch_size)

				#print("total_batch: ", total_batch)
				for i in range(total_batch) :

					row_size = self.Dataset.batch_end - self.Dataset.batch_beg

					batch_x = self.Dataset.next_batch()
					batch_mask, batch_mask_inverse = self.Dataset.next_batch_mask(row_size, self.missing_perc)
					self.Dataset.inc_batch()

					#print(batch_x.shape, batch_mask.shape)
					corrupted_batch = np.asarray(batch_x)*np.asarray(batch_mask)

					#print("X: ",type(batch_x)," ",batch_x.shape,"\n\n")
					#print("y: ",type(batch_y)," ",batch_y.shape,"\n\n\n")



					_, l, recons_batch = sess.run([self.optimizer, self.loss, recons_X], feed_dict = {self.X : batch_x, self.input_X : corrupted_batch, self.X_mask : batch_mask, self.X_mask_inverse : batch_mask_inverse})
					
					
					if epoch == (self.epochs-1) :
						full_recons_matrix = self.Dataset.compile_batches(recons_batch, full_recons_matrix)
						#print("full_recons_matrix.shape: ", full_recons_matrix.shape)


				 

				print("Epoch: ", epoch + 1, "cost: ", "{:.5}".format(l))

		self.Dataset.save_matrix(results_filePath, full_recons_matrix)




def main() :

	#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	Dataset = DataHandler('./Experiment/BioDataset1', mask_given = False)
	model = DAPL(Dataset = Dataset, learning_rate = 0.001 , epochs = 10 , batch_size = 150, shape = (None,1200), missing_perc = 0.01)

	model.network_weights_biases([1200,600,300,600,1200])
	model.train(save_results = True)

main()







