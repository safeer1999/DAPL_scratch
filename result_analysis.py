import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd 
import scipy.sparse as scp 
import argparse
from math import sqrt
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument('--dataset_type', default = 'R')
parser.add_argument("--recons")
parser.add_argument("--mask")
parser.add_argument('--result_file_path' , type = str , default = './output_results.csv')

args = parser.parse_args()


def split(dataset,val_perc = 0.1, test_perc = 0.2):
	
	val_size = int(val_perc*dataset.shape[0])
	test_size = int(test_perc*dataset.shape[0])
	train_size = dataset.shape[0] - (val_size+test_size)

	train_set = dataset[:train_size,:]
	val_set = dataset[train_size : train_size + val_size, :]
	test_set = dataset[train_size + val_size : , :]

	return train_set, val_set, test_set


dataset =  scp.load_npz(args.dataset).todense()
train_set, val_set, test_set = split(dataset)

if args.dataset_type == 'train' :
	R = train_set

elif args.dataset_type == 'test' :
	R = test_set

else :
	R = dataset


recons = np.load(args.recons)
mask = np.load(args.mask)

R_masked = np.squeeze(np.asarray(R[mask == 0]))
recons_masked = recons[mask == 0]

correl = np.corrcoef(R_masked, recons_masked)[0,1]
RMSE = sqrt(mean_squared_error(R_masked, recons_masked))

df_results = pd.DataFrame({'R' : list(R_masked), 'recons' : list(recons_masked) , 'Correlation' : [correl]*R_masked.shape[0], 'RMSE' : [RMSE]*R_masked.shape[0]})

df_results.iloc[1:, [2,3]] = np.nan

df_results.to_csv(args.result_file_path)


