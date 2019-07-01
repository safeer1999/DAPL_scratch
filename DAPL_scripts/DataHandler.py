import tensorflow as tf 
import numpy as np
import pandas as pd 
import scipy.sparse as scp 


#Used to manage datasets and result data

class DataHandler :

	def __init__(self, directory_path, mask_given  = False):
		self.R = scp.load_npz(directory_path).todense()
		self.mask_given = mask_given
		if self.mask_given :
			self.mask = scp.load_npz(directory_path +  '/' + 'train_mask.npz').todense()
		else :
			self.mask = None


	def split(self,dataset, val_perc = 0.1, test_perc = 0.2):
		
		val_size = int(val_perc*dataset.shape[0])
		test_size = int(test_perc*dataset.shape[0])
		train_size = dataset.shape[0] - (val_size+test_size)

		train_set = dataset[:train_size,:]
		val_set = dataset[train_size : train_size + val_size, :]
		test_set = dataset[train_size + val_size : , :]

		return train_set, val_set, test_set


	def get_mask_from_dataset(self, R) :

	    mask = np.where(R == 0 , False, True)
	    mask_inverse = mask == False

	    return mask,mask_inverse

	#-----------------------------------------BATCH CONTROL----------------------------


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

	def next_batch_mask(self, batch_beg, batch_end, row_size = 0, missing_perc = 0.1, dataset = None) :#produce batch for mask matrix or produces a random generated batch using missing_perc

		if self.mask_given :
			batch_mask = dataset[batch_beg:batch_end, :]
			batch_mask_inverse = np.where(batch_mask , 0 , 1)
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
	
	#------------------------------EVALUATION------------------------------------------

	def correl(self, y, y_pred):
		
		y = np.squeeze(np.asarray(y))
		y_pred = np.squeeze(np.asarray(y_pred))

		correl_val = np.corrcoef(y, y_pred)[0,1]

		return correl_val