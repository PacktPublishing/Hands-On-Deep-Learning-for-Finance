# create datasets and save them to hdf5 format

import numpy as np 
import h5py
from tensorflow_setup import *

def savedataset(csvfilename, h5filename):

	dt = h5py.special_dtype(vlen=str)
	hf = h5py.File('h5files/'+ h5filename, 'w')

	for i in range(0,len(csvfilename)):
		csvfile = csvfilename[i]
		if(csvfile[:4] != 'test'):
			X_tr, X_vld, lab_tr, lab_vld, list_ch_train = create_tensorflow_train_data(csvfile)
			hf.create_dataset('X_tr',data=X_tr)
			hf.create_dataset('X_vld',data=X_vld)
			hf.create_dataset('lab_tr',data=lab_tr)
			hf.create_dataset('lab_vld',data=lab_vld)
			hf.create_dataset('list_ch_train',data=list_ch_train, dtype = dt)
			del(X_tr, X_vld, lab_tr, lab_vld, list_ch_train)
		else:
			X_test, y_test, list_ch_test = create_tensorflow_test_data(csvfile)
			hf.create_dataset('X_test', data = X_test)
			hf.create_dataset('y_test', data = y_test)
			hf.create_dataset('list_ch_test', data = list_ch_test, dtype = dt)
			del(X_test, y_test, list_ch_test)

	hf.close()


savedataset(['test_50.csv', 'train_50.csv'], 'hdf_50.h5')

savedataset(['test_100.csv', 'train_250.csv'], 'hdf_250.h5')

savedataset(['test_150.csv', 'train_500.csv'], 'hdf_500.h5')

savedataset(['test_250.csv', 'train_750.csv'], 'hdf_750.h5')

savedataset(['test_500.csv', 'train_1000.csv'], 'hdf_1000.h5')


