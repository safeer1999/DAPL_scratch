from DAPL_base import DAPL 
from DAPL_base import DataHandler
import tensorflow as tf 
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

def test_func(model, test_set, test_mask):

	model.saver = tf.train.Saver()
	loss = None
	recons = None

	with tf.Session() as sess :

		model.saver.restore(sess,'./saved_model/')

		test_mask_inverse = np.where(test_mask , 0 , 1)

		corrupted_set = np.asarray(test_set)*np.asarray(test_mask)

		loss,recons = sess.run([model.loss, model.recons_X], feed_dict = {model.X : test_set, model.input_X : corrupted_set, model.X_mask_inverse : test_mask})

		return loss, recons






def main() :

	#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	Dataset = DataHandler(args.input_file, mask_given = args.set_mask)
	_,_,test_set = Dataset.split(Dataset.R)
	_,_,test_mask = Dataset.split(Dataset.mask)

	model = DAPL(Dataset = Dataset, learning_rate = args.lr , epochs = args.epochs , missing_perc = args.missing_perc)
	model.netBuild(featureNum = Dataset.R.shape[1])
	model.define_network()
	#model.train(save_results = args.save_results, results_filePath = args.output_filePath, mask_filePath = args.output_filePath, batch_size = args.batch_size)

	loss, recons = test_func(model, test_set, test_mask)

	print("Loss: ", loss)

if __name__ == '__main__':
	main()
