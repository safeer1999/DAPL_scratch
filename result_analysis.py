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
parser.add_argument("--recons")
parser.add_argument("--mask")

args = parser.parse_args()

R = scp.load_npz(args.dataset).todense()
recons = np.load(args.recons)
mask = np.load(args.mask)

R_masked = np.squeeze(np.asarray(R[mask == 0]))
recons_masked = recons[mask == 0]

correl = np.corrcoef(R_masked, recons_masked)[0,1]
RMSE = sqrt(mean_squared_error(R_masked, recons_masked))


