import tensorflow as tf 
import numpy as np
import pandas as pd 
import scipy.sparse as scp 

from DataHandler import DataHandler


class DAPL :

	def __init__(self,learning_rate = 0.1 , epochs = 10 , batch_size = 20, missing_perc = 0.1, shape = (0,0)) :


		#Parameters
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.shape = shape # to be inititialized in while network building
		self.missing_perc = missing_perc


		#Training Paceholders
		self.X = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.X_mask = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.X_mask_inverse = tf.placeholder(tf.float32, [None, self.shape[1]])
		self.input_X = tf.placeholder(tf.float32, [None, self.shape[1]])

		#Session variables
		self.sess = None
		self.init_op = tf.global_variables_initializer()

		#Saving Model
		self.saver = None


#----------------------------------------------NEURAL NETWORK--------------------------------------------------------------------------


	def netBuild(self,featureNum = 0 , reduct_fact = 2, numLayers = 2, load_model_bool = False) :

	    network = []

	    numNodes = featureNum

	    for i in range(numLayers) :

	        network.append(numNodes)

	        numNodes = numNodes//2

	    for i in range(numLayers) :
	        network.append(numNodes)
	        numNodes = numNodes*2

	    network.append(featureNum)

	    if load_model_bool :
	    	self.restore_network_weights_biases(network)

	    else :
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

	def define_network(self) :

		self.recons_X = self.network_func()
		self.loss_func(self.X, self.recons_X)
		self.optimizer_func(tf.train.AdamOptimizer)


#----------------------------------------------------------------------------------------------------------------------------------------

	def train(self, Dataset , save_results = False, results_filePath = '.', mask_filePath = '.', batch_size = None, save_model_bool = False, model_dir = None) : 


		#Saving the model
		self.saver = tf.train.Saver()

		#Split dataset into training validation and testing
		train_set, val_set, test_set = Dataset.split(Dataset.R)

		if Dataset.mask_given :
			train_mask, val_mask, test_mask = Dataset.split(Dataset.mask)
			test_mask_inverse = np.where(test_mask , 0 , 1)

		else :
			train_mask = None
			val_mask = None
			test_mask_inverse = np.random.binomial(1, self.missing_perc, size=test_set.shape[0]*test_set.shape[1]).reshape(test_set.shape[0], test_set.shape[1]) #abstract random mask creation
			test_mask = np.where(test_mask_inverse , 0 , 1)

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
				batch_beg_train, batch_end_train, total_batch_train = Dataset.batch_init(dataset = train_set, batch_size = train_batch_size) #for training
				batch_beg_val, batch_end_val, total_batch_val = Dataset.batch_init(dataset = train_set, batch_size = val_batch_size) #for validation

				#print("total_batch: ", total_batch)
				for i in range(total_batch_train) :

					#-----------------------------------------TRAINING--------------------------------------

					row_size = batch_end_train - batch_beg_train

					batch_x = Dataset.next_batch(train_set, batch_beg_train, batch_end_train)
					batch_mask, batch_mask_inverse = Dataset.next_batch_mask(batch_beg_train, batch_end_train, row_size, self.missing_perc, dataset = train_mask)
					batch_beg_train, batch_end_train = Dataset.inc_batch(batch_beg_train, batch_end_train, train_batch_size)

					#print('training: ', batch_x.shape, batch_mask.shape)
					corrupted_batch = np.asarray(batch_x)*np.asarray(batch_mask)

					#print("X: ",type(batch_x)," ",batch_x.shape,"\n\n")
					#print("y: ",type(batch_y)," ",batch_y.shape,"\n\n\n")



					_, l, recons_batch = sess.run([self.optimizer, self.loss, self.recons_X], feed_dict = {self.X : batch_x, self.input_X : corrupted_batch, self.X_mask : batch_mask, self.X_mask_inverse : batch_mask_inverse})
					#-----------------------------------------------------------------------------------------

					#-----------------------------------------VALIDATION--------------------------------------

					row_size_val = batch_end_val - batch_beg_val

					batch_x_val = Dataset.next_batch(val_set, batch_beg_val, batch_end_val)
					batch_mask_val, batch_mask_inverse_val = Dataset.next_batch_mask(batch_beg_val, batch_end_val, row_size_val, self.missing_perc,dataset = val_mask)
					batch_beg_val, batch_end_val = Dataset.inc_batch(batch_beg_val, batch_end_val, val_batch_size)

					#print('validation: ', batch_x_val.shape, batch_mask_val.shape)
					corrupted_batch_val = np.asarray(batch_x_val)*np.asarray(batch_mask_val)

					#print("X: ",type(batch_x)," ",batch_x.shape,"\n\n")
					#print("y: ",type(batch_y)," ",batch_y.shape,"\n\n\n")

					l_val,recons_batch_val = sess.run([self.loss, self.recons_X], feed_dict = {self.X : batch_x_val, self.input_X : corrupted_batch_val, self.X_mask_inverse : batch_mask_inverse_val})


					#------------------------------------------------------------------------------------------
					
					if epoch == (self.epochs-1) :
						full_recons_matrix = Dataset.compile_batches(recons_batch, full_recons_matrix)
						full_mask_matrix = Dataset.compile_batches(batch_mask, full_mask_matrix)
						#print("full_recons_matrix.shape: ", full_recons_matrix.shape)			
				 

				print("Epoch: ", epoch + 1, "\ncost: ", "{:.5}".format(l))
				print('cost_val: ',"{:.5}".format(l_val))
				print('\n')



			if save_model_bool :
				self.save_model(sess, model_dir)

			#test_loss,test_recons= self.test(test_set,sess, test_mask, test_mask_inverse)



		if save_results :
			Dataset.save_matrix(results_filePath + '/recons', full_recons_matrix)
			Dataset.save_matrix(mask_filePath +'/mask', full_mask_matrix)
			Dataset.save_matrix(results_filePath + '/test', test_recons)
			Dataset.save_matrix(mask_filePath + '/test_mask',test_mask)

		#print("Test Loss :", test_loss)



	def test(self,dataset,sess, test_mask, test_mask_inverse):

		loss = None
		recons = None

		#print('\ntest dataset :', dataset.shape, test_mask.shape,'\n')

		#test_mask_inverse = np.random.binomial(1, self.missing_perc, size=dataset.shape[0]*dataset.shape[1]).reshape(dataset.shape[0], dataset.shape[1])
		#test_mask = np.where(test_mask_inverse , 0 , 1)

		corrupted_set = np.asarray(dataset)*np.asarray(test_mask)

		loss,recons = sess.run([self.loss, self.recons_X], feed_dict = {self.X : dataset, self.input_X : corrupted_set, self.X_mask_inverse : test_mask})

		return loss, recons

	#------------------------------------------------------------SAVE AND RESTORE MODEL--------------------------------------------------

	def save_model(self,sess, model_dir = None) :

		self.saver.save(sess, model_dir)

	def restore_model(self,sess,model_dir) :

		tf.reset_default_graph()
		loader = tf.train.import_meta_graph(model_dir + '.meta')
		self.graph = tf.get_default_graph()

		loader.restore(sess,model_dir)


		self.graph = tf.get_default_graph()

	def restore_network_weights_biases(self,num_nodes) :

		self.weights = []
		self.biases = []

		#initializer=tf.variance_scaling_initializer()

			
		for i in range(len(num_nodes)-1) :

			axis_0 = num_nodes[i]
			axis_1 = num_nodes[i+1]

			#print('-----------------------------------------------------------')
			#print('axis_0: ', axis_0,"  axis_1: ", axis_1)
			#print('-----------------------------------------------------------')

			W = self.graph.get_tensor_by_name('W' + str(i) + ':0')
			b = self.graph.get_tensor_by_name('b' + str(i) + ':0')

			self.weights.append(W)
			self.biases.append(b)



	#------------------------------------------------------------------------------------------------------------------------------------