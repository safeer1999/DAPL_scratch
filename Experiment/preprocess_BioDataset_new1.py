#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:31:51 2019

@author: safeer
"""
#import libraries
import pandas as pd
import numpy as np
import scipy.sparse as scp
import random
import os
import sys

#importing datasets from files
original = pd.read_csv("../Bio_Dataset_text/GROUND_TRUTH1200.txt",sep = ' ')

#adding columns
columns = [str("snip"+str(i)) for i in range(8955)]
original.columns = columns

#random_columns = [int(random.random()*8955) for i in range(1200)]

#1200x1200 from orginal where columns are random
X = original.values[:,1:]
if sys.argv[3] != 'all' : #sys.argv[3] -> number of columns to include if 'all' complete dataset
	X = X[:,:int(sys.argv[3])]
	

#all valuues are shifted by one as 0 values in this dataset carries information {0,1,2} => {1,2,3}
X_shift1 = np.asarray(np.add(X,1), dtype = 'float64')


#creating the modified dataset
def create_dataset(mask_cycle, append_cycle = 20, experiment_dir = None, view_mask = None) : #mask_cycle-> which column to mask and append_cycle -> which iteration to append the dataset to the total
#append_cycle = 20
#mask_cycle = 100
    train = np.empty(shape = (1,X.shape[1]))
    train_mask = np.empty(shape = (1,X.shape[1]))
    
    for i in range(0,100,append_cycle) :
        
        mask  = np.asarray(np.full((1199,X.shape[1]), True))
        
        #for j in range(0,X.shape[1]-(154-mask_cycle),mask_cycle) :
        for j in range(0,X.shape[1],mask_cycle) :
            if (i+j) < X.shape[1] :
                mask[:,i+j] = False
            else :
                break
        
        #new_set = np.multiply(original, mask)
        
        train = np.append(train, X_shift1, axis = 0)
        train_mask = np.append(train_mask, mask, axis = 0)
        print(train.shape)
    
    train = np.delete(train,0,0)
    train_mask = np.delete(train_mask,0,0)
    
    
    train = np.asarray(train, dtype = np.float64)
    
    #missing percentage
    total = train.shape[0]*train.shape[1]
    nonmissing = np.count_nonzero(train_mask)
    missing_perc = (1 - ((nonmissing*1.0)/total)) * 100
    
    #save the datasets
    if not os.path.exists("../"  + experiment_dir) :
        os.mkdir("../" + experiment_dir)    

    dir_name = str("../" + experiment_dir + "/")  + str("BioDataset") + str(int(missing_perc))
    if not os.path.exists(dir_name) :
        os.mkdir(dir_name)
    
    #view_mask = train_mask
    #pd.DataFrame(train_mask).to_csv('view_mask.csv')
    
    sparse_train_mask = scp.csc_matrix(train_mask)
    sparse_train = scp.csc_matrix(np.asarray(train))
    
    scp.save_npz(dir_name + "/rating",sparse_train)
    scp.save_npz(dir_name + "/train_mask",sparse_train_mask)
    scp.save_npz(dir_name + "/val_mask",sparse_train_mask)


#parameters = [100,50,25,10,5,2,1]
parameters = [100]
for i in parameters :
    create_dataset(i,int(sys.argv[1]), sys.argv[2]) #command line arguments - arg1 represents -> 100/arg1 time the datasets should be re-appended
    #                                               arg2 represent the directory to store the dataset leave it as null to store in present directory
    #create_dataset(i, 20, 'Experiment', view_mask)

