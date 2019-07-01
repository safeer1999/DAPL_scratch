import numpy as np
import scipy.sparse as scp


for i in range(10,91,10) :
	
	R = scp.load_npz('Experiment' + str(i) + '/BioDataset1/rating.npz').todense()
	train = R[0:841,:]
	val = R[841:960,:]

	np.save('Experiment' + str(i) + '/results/train',train)
	np.save('Experiment' + str(i) + '/results/val',val)
	
