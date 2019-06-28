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
parser.add_argument("--model_dir","trained_model")

args = parser.parse_args()


#-----------------------------------------------------------------------------------------------



def main() :
	Dataset = DataHandler(args.input_file, mask_given = args.set_mask)

	model = DAPL(shape = Dataset.R.shape)
	model.restore_graph(args.model_dir)
	model.init_tensors()
	model.restore_model(model.sess,args.model_dir)
	model.netBuild(featureNum = Dataset.R.shape[1], load_model_bool = True)
	model.define_network(mode = 'test')	


	X = Dataset.R
	mask, mask_inverse = Dataset.get_mask_from_dataset(X)


	loss,recons_X = model.test(X,model.sess,mask, mask_inverse )

	X_masked = X[mask == 0]
	recons_masked = recons_X[mask == 0]
	acc = Dataset.correl(X_masked, recons_masked)

	if args.save_results :
		pd.DataFrame(recons_X).to_csv(args.output_filePath + '/recons.csv')
		pd.DataFrame(X).to_csv(args.output_filePath + '/X.csv')


	imputed_set = np.add(X ,np.multiply(mask_inverse,recons_X))
	pd.DataFrame(imputed_set).to_csv(args.output_filePath + '/imputed_set.csv')		

	#print("Loss: ", loss)
	#print("Accuracy: ", acc)



main()