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
#parser.add_argument("--lr", type = float, default = 0.001)
#parser.add_argument("--epochs", type = int, default = 10)
#parser.add_argument("--batch_size", type = int , default = 100)
#parser.add_argument("--shape" , default  = '0,0' )
#parser.add_argument("--missing_perc", type = float , default = 1)
parser.add_argument("--save_results", type = bool, default = False)
parser.add_argument("--output_filePath")
#parser.add_argument("--save_model", type = bool, default = False)
parser.add_argument("--model_dir")

args = parser.parse_args()


#-----------------------------------------------------------------------------------------------



def main() :
	Dataset = DataHandler(args.input_file, mask_given = args.set_mask)

	model = DAPL(shape = Dataset.R.shape)
	model.sess = tf.Session()
	model.restore_model(model.sess,args.model_dir)
	model.netBuild(featureNum = Dataset.R.shape, load_model_bool = True)
	model.define_network()	

	#_,_,test_set = Dataset.split(Dataset.R)
	#test_mask_inverse = np.random.binomial(1, 0.05, size=test_set.shape[0]*test_set.shape[1]).reshape(test_set.shape[0], test_set.shape[1]) #abstract random mask creation
	#test_mask = np.where(test_mask_inverse , 0 , 1)

	#loss,_ = model.test(test_set,model.sess,test_mask, test_mask_inverse )

	#print("Loss: ", loss)



main()