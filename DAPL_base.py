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

#modification of cmd args
args.shape = tuple(list(map(int, args.shape.split(',')))) 
args.missing_perc/=100.0
#-----------------------------------------------------------------------------------------------



def main() :

	#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	Dataset = DataHandler(args.input_file, mask_given = args.set_mask)

	model = DAPL(learning_rate = args.lr , epochs = args.epochs , missing_perc = args.missing_perc, shape = Dataset.R.shape)
	model.init_tensors()
	model.netBuild(featureNum = Dataset.R.shape[1])
	model.define_network()
	model.train(Dataset, save_results = args.save_results, results_filePath = args.output_filePath, mask_filePath = args.output_filePath, batch_size = args.batch_size, save_model_bool = args.save_model, model_dir = args.model_dir)

if __name__ == '__main__':
	main()



