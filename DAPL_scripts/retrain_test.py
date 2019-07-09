import tensorflow as tf 
import numpy as np
import pandas as pd 
import scipy.sparse as scp 
import argparse

from DAPL import DAPL
from DataHandler import DataHandler


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
parser.add_argument("--save_model", type = bool, default = False)
parser.add_argument("--model_dir")

args = parser.parse_args()

#-----------------------------------------------------------------------------------------------


def main_test() :

	Dataset = DataHandler(args.input_file, mask_given = args.set_mask)

	model = DAPL(shape = Dataset.R.shape)


	model.init_tensors()
	model.netBuild(featureNum = Dataset.R.shape[1], load_model_bool = True)
	model.set_loader()
	model.restore_model(model.sess,args.model_dir)
	model.define_network(mode = 'train')
	
	'''
	#print("-----------------------------------------------------------------")
	for op in tf.get_default_graph().get_operations():
		print(str(op.name))

	#print("-----------------------------------------------------------------")
	'''

	_,_,test_set = Dataset.split(Dataset.R)
	test_mask_inverse = np.random.binomial(1, 0.05, size=test_set.shape[0]*test_set.shape[1]).reshape(test_set.shape[0], test_set.shape[1]) #abstract random mask creation
	test_mask = np.where(test_mask_inverse , 0 , 1)

	loss,recons_test = model.test(test_set,model.sess,test_mask, test_mask_inverse )

	test_set_masked = test_set[test_mask == 0]
	recons_test_masked = recons_test[test_mask == 0]
	acc = Dataset.correl(test_set_masked, recons_test_masked)

	print("Loss: ", loss)
	print("Accuracy: ", acc)



def main_train() :

	Dataset = DataHandler(args.input_file, mask_given = args.set_mask)

	model = DAPL(shape = Dataset.R.shape)


	model.init_tensors()
	model.netBuild(featureNum = Dataset.R.shape[1], load_model_bool = True)
	model.set_loader()
	model.restore_model(model.sess,args.model_dir)
	model.define_network(mode = 'train')


	
	model.train(Dataset, save_results = args.save_results, results_filePath = args.output_filePath, mask_filePath = args.output_filePath, batch_size = args.batch_size, save_model_bool = args.save_model, model_dir = args.model_dir)


#main_test()
main_train()
