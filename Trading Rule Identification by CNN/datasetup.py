import h5py

def get_tf_train_data(h5filename):
hf = h5py.File('h5files/' + h5filename, 'r')
 X_tr = hf['X_tr'][:]
 X_vld = hf['X_vld'][:]
 lab_tr = hf['lab_tr'][:]
 lab_vld = hf['lab_vld'][:]
 list_ch_train = hf['list_ch_train'][:]
 hf.close()
 return X_tr, X_vld, lab_tr, lab_vld, list_ch_train

def get_tf_test_data(h5filename):
hf = h5py.File('h5files/' + h5filename, 'r')
 X_test = hf['X_test'][:]
 y_test = hf['y_test'][:]
 list_ch_test = hf['list_ch_test'][:]
return X_test, y_test, list_ch_test
