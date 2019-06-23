import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd 
import scipy.sparse as scp 
import argparse

#--------------------------------------Command Line Argument------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_file")
parser.add_argument("--set_mask", type = bool, default = False)
parser.add_argument("--lr", type = float, default = 0.001)
parser.add_argument("--epochs", type = int, default = 10)
parser.add_argument("--batch_size", type = int , default = 100)
parser.add_argument("--shape" , default  = '0,0' )
parser.add_argument("--missing_perc", type = float , default = 1)
parser.add_argument("--save_results", type = bool, default = False)
parser.add_argument("--output_filePath")

args = parser.parse_args()

#modification of cmd args
args.shape = tuple(list(map(int, args.shape.split(',')))) 
args.missing_perc/=100.0
#-----------------------------------------------------------------------------------------------


#Used to manage datasets and result data
class DataHandler :

	def __init__(self, directory_path, mask_given  = False):
		self.R = scp.load_npz(directory_path +  '/' + 'rating.npz').todense()
		self.mask_given = mask_given
		if self.mask_given :
			self.mask = scp.load_npz(directory_path +  '/' + 'train_mask.npz').todense()
		else :
			self.mask = None


	def split(self,val_perc = 0.1, test_perc = 0.2):
		
		val_size = int(val_perc*self.R.shape[0])
		test_size = int(test_perc*self.R.shape[0])
		train_size = self.R.shape[0] - (val_size+test_size)

		train_set = self.R[:train_size,:]
		val_set = self.R[train_size : train_size + val_size, :]
		test_set = self.R[train_size + val_size : , :]

		return train_set, val_set, test_set

	#-----------------------------------------BATCH CONTROL----------------------------

	'''def batch_init(self, batch_size) : #initialize dataset batch control variables
		self.batch_beg = 0
		self.batch_end = 0
		self.batch_size = 0

		self.batch_end += self.R.shape[0]%batch_size
		self.batch_size  = batch_size

		num_batches = self.get_num_batch()

		return num_batches'''

	def batch_init(self, dataset, batch_size) : #initialize dataset batch control variables
		batch_beg = 0
		batch_end = 0

		batch_end += dataset.shape[0]%batch_size

		num_batches = self.get_num_batch(dataset, batch_size)

		return batch_beg, batch_end, num_batches



	def next_batch(self, dataset, batch_beg, batch_end):# produces the next batch of rows from the datasets sequentially as when its called

		batch_R = dataset[batch_beg:batch_end, :]


		#print("Batch R:", batch_R.shape)

		return batch_R

	def next_batch_mask(self, batch_beg, batch_end, row_size = 0, missing_perc = 0.1) :#produce batch for mask matrix or produces a random generated batch using missing_perc

		if self.mask_given :
			batch_mask = self.mask[batch_beg:batch_end, :]
			#print('batch_mask.shape: ',batch_mask.shape)

		else :
			#print('row_size', row_size)
			batch_mask_inverse = np.random.binomial(1, missing_perc, size=row_size*self.R.shape[1]).reshape(row_size, self.R.shape[1])

		batch_mask = np.where(batch_mask_inverse , 0 , 1)


		return batch_mask,batch_mask_inverse

	def inc_batch(self, batch_beg, batch_end, batch_size) : #used for iterating through the datatset batchwise

		#print('self.batch_beg', self.batch_beg)
		#print('self.batch_end', self.batch_end)

		batch_beg = batch_end
		batch_end+=batch_size

		return batch_beg, batch_end

	def get_num_batch(self, dataset, batch_size) :#returns no of batches in the dataset

		num_batches = dataset.shape[0]//batch_size

		num_final_batch = dataset.shape[0] - num_batches*batch_size #number of instances in the final batch if the row size is not a perfect multiple of batch_size

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

	def save_to_file() :
		pass


	#----------------------------------------------------------------------------------




class DAPL :

	def __init__(self, Dataset = None , learning_rate = 0.1 , epochs = 10 , batch_size = 20, missing_perc = 0.1) :

		#Datasets
		self.Dataset = Dataset

		#Parameters
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.shape = Dataset.R.shape
		self.missing_perc = missing_perc


		#Training Paceholders
		self.X = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.X_mask = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.X_mask_inverse = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.input_X = tf.placeholder(tf.float32, [None, self.shape[1]])

		#Session variables
		self.sess = None
		self.init_op = tf.global_variables_initializer()


#----------------------------------------------NEURAL NETWORK--------------------------------------------------------------------------


	def netBuild(self,featureNum = 0 , reduct_fact = 2, numLayers = 2) :

	    network = []

	    numNodes = featureNum

	    for i in range(numLayers) :

	        network.append(numNodes)

	        numNodes = numNodes//2

	    for i in range(numLayers) :
	        network.append(numNodes)
	        numNodes = numNodes*2

	    network.append(featureNum)

	    self.network_weights_biases(network)


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


#----------------------------------------------------------------------------------------------------------------------------------------

	def train(self, save_results = False, results_filePath = '.', mask_filePath = '.', batch_size = None) : 

		self.recons_X = self.network_func()
		self.loss_func(self.X, self.recons_X)
		self.optimizer_func(tf.train.AdamOptimizer)

		train_set, val_set, test_set = self.Dataset.split()

		full_recons_matrix = np.empty(shape = (0,self.shape[1]))
		full_mask_matrix = np.empty(shape = (0,self.shape[1]))

		init_op = tf.global_variables_initializer()

		with tf.Session() as sess :

			sess.run(init_op)

			for epoch in range(self.epochs) :

				l = 0
				#batch control variables
				train_batch_size = batch_size
				val_batch_size = int(batch_size*(val_set.shape[0]*1.0 / train_set.shape[0]))
				batch_beg_train, batch_end_train, total_batch_train = self.Dataset.batch_init(dataset = train_set, batch_size = train_batch_size) #for training
				batch_beg_val, batch_end_val, total_batch_val = self.Dataset.batch_init(dataset = train_set, batch_size = val_batch_size) #for validation

				#print("total_batch: ", total_batch)
				for i in range(total_batch_train) :

					#-----------------------------------------TRAINING--------------------------------------

					row_size = batch_end_train - batch_beg_train

					batch_x = self.Dataset.next_batch(train_set, batch_beg_train, batch_end_train)
					batch_mask, batch_mask_inverse = self.Dataset.next_batch_mask(batch_beg_train, batch_end_train, row_size, self.missing_perc)
					batch_beg_train, batch_end_train = self.Dataset.inc_batch(batch_beg_train, batch_end_train, train_batch_size)

					#print('training: ', batch_x.shape, batch_mask.shape)
					corrupted_batch = np.asarray(batch_x)*np.asarray(batch_mask)

					#print("X: ",type(batch_x)," ",batch_x.shape,"\n\n")
					#print("y: ",type(batch_y)," ",batch_y.shape,"\n\n\n")



					_, l, recons_batch = sess.run([self.optimizer, self.loss, self.recons_X], feed_dict = {self.X : batch_x, self.input_X : corrupted_batch, self.X_mask : batch_mask, self.X_mask_inverse : batch_mask_inverse})
					#-----------------------------------------------------------------------------------------

					#-----------------------------------------VALIDATION--------------------------------------

					row_size_val = batch_end_val - batch_beg_val

					batch_x_val = self.Dataset.next_batch(val_set, batch_beg_val, batch_end_val)
					batch_mask_val, batch_mask_inverse_val = self.Dataset.next_batch_mask(batch_beg_val, batch_end_val, row_size_val, self.missing_perc)
					batch_beg_val, batch_end_val = self.Dataset.inc_batch(batch_beg_val, batch_end_val, val_batch_size)

					#print('validation: ', batch_x_val.shape, batch_mask_val.shape)
					corrupted_batch_val = np.asarray(batch_x_val)*np.asarray(batch_mask_val)

					#print("X: ",type(batch_x)," ",batch_x.shape,"\n\n")
					#print("y: ",type(batch_y)," ",batch_y.shape,"\n\n\n")

					l_val,recons_batch_val = sess.run([self.loss, self.recons_X], feed_dict = {self.X : batch_x_val, self.input_X : corrupted_batch_val, self.X_mask_inverse : batch_mask_inverse_val})


					#------------------------------------------------------------------------------------------
					
					if epoch == (self.epochs-1) :
						full_recons_matrix = self.Dataset.compile_batches(recons_batch, full_recons_matrix)
						full_mask_matrix = self.Dataset.compile_batches(batch_mask, full_mask_matrix)
						#print("full_recons_matrix.shape: ", full_recons_matrix.shape)


				 

				print("Epoch: ", epoch + 1, "\ncost: ", "{:.5}".format(l))
				print('cost_val: ',"{:.5}".format(l_val))
				print('\n')


			test_loss,test_recons, test_mask = self.test(test_set,sess)


		if save_results :
			self.Dataset.save_matrix(results_filePath + '/recons', full_recons_matrix)
			self.Dataset.save_matrix(mask_filePath +'/mask', full_mask_matrix)
			self.Dataset.save_matrix(results_filePath + '/test', test_recons)
			self.Dataset.save_matrix(mask_filePath + '/test_mask',test_mask)

		print("Test Loss :", test_loss)



	def test(self,dataset,sess):

		loss = None
		recons = None

		test_mask_inverse = np.random.binomial(1, self.missing_perc, size=dataset.shape[0]*dataset.shape[1]).reshape(dataset.shape[0], dataset.shape[1])
		test_mask = np.where(test_mask_inverse , 0 , 1)

		corrupted_set = np.asarray(dataset)*np.asarray(test_mask)

		loss,recons = sess.run([self.loss, self.recons_X], feed_dict = {self.X : dataset, self.input_X : corrupted_set, self.X_mask_inverse : test_mask})

		return loss, recons, test_mask








def main() :

	#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	Dataset = DataHandler(args.input_file, mask_given = args.set_mask)

	model = DAPL(Dataset = Dataset, learning_rate = args.lr , epochs = args.epochs , missing_perc = args.missing_perc)
	model.netBuild(featureNum = Dataset.R.shape[1])
	model.train(save_results = args.save_results, results_filePath = args.output_filePath, mask_filePath = args.output_filePath, batch_size = args.batch_size)

main()







