# diagnostics

from tensorflow_setup import *
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# diagnostics for trading rule

# read the results file and put it in the format recognized by heatmap
hf = h5py.File('h5files/simpleclass_train_50.h5', 'r')
predict_val = hf['predict_valid'][:]
ytest = hf['y_test'][:]
x = np.argmax(predict_val, axis = 1)
predict_valid = np.zeros(predict_val.shape)
predict_valid[x == 0,0] = 1
predict_valid[x == 1,1] = 1
predict_valid[x == 2,2] = 1

df = prediction_accuracy(ytest, predict_valid)

ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')

# now save the heatmap
ax.figure.savefig('figures/simpleclass_50_50.png')
plt.clf()


hf = h5py.File('h5files/simpleclass_train_250_test_100.h5', 'r')
predict_val = hf['predict_valid'][:]
ytest = hf['y_test'][:]
x = np.argmax(predict_val, axis = 1)
predict_valid = np.zeros(predict_val.shape)
predict_valid[x == 0,0] = 1
predict_valid[x == 1,1] = 1
predict_valid[x == 2,2] = 1

df = prediction_accuracy(ytest, predict_valid)

ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')

# now save the heatmap
ax.figure.savefig('figures/simpleclass_250_100.png')
plt.clf()


# now for cnn files

hf = h5py.File('h5files/cnn_train_50.h5', 'r')
predict_val = hf['predict_valid'][:]
ytest = hf['y_test'][:]
x = np.argmax(predict_val, axis = 1)
predict_valid = np.zeros(predict_val.shape)
predict_valid[x == 0,0] = 1
predict_valid[x == 1,1] = 1
predict_valid[x == 2,2] = 1

df = prediction_accuracy(ytest, predict_valid)

ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')

# now save the heatmap
ax.figure.savefig('figures/cnn_train_50.png')
plt.clf()

hf = h5py.File('h5files/cnn_train_250.h5', 'r')
predict_val = hf['predict_valid'][:]
ytest = hf['y_test'][:]
x = np.argmax(predict_val, axis = 1)
predict_valid = np.zeros(predict_val.shape)
predict_valid[x == 0,0] = 1
predict_valid[x == 1,1] = 1
predict_valid[x == 2,2] = 1

df = prediction_accuracy(ytest, predict_valid)

ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')

# now save the heatmap
ax.figure.savefig('figures/cnn_train_250.png')
plt.clf()

hf = h5py.File('h5files/cnn_train_500.h5', 'r')
predict_val = hf['predict_valid'][:]
ytest = hf['y_test'][:]
x = np.argmax(predict_val, axis = 1)
predict_valid = np.zeros(predict_val.shape)
predict_valid[x == 0,0] = 1
predict_valid[x == 1,1] = 1
predict_valid[x == 2,2] = 1

df = prediction_accuracy(ytest, predict_valid)

ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')

# now save the heatmap
ax.figure.savefig('figures/cnn_train_500.png')
plt.clf()

hf = h5py.File('h5files/cnn_train_750.h5', 'r')
predict_val = hf['predict_valid'][:]
ytest = hf['y_test'][:]
x = np.argmax(predict_val, axis = 1)
predict_valid = np.zeros(predict_val.shape)
predict_valid[x == 0,0] = 1
predict_valid[x == 1,1] = 1
predict_valid[x == 2,2] = 1

df = prediction_accuracy(ytest, predict_valid)

ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')

# now save the heatmap
ax.figure.savefig('figures/cnn_train_750.png')
plt.clf()

hf = h5py.File('h5files/cnn_train_1000.h5', 'r')
predict_val = hf['predict_valid'][:]
ytest = hf['y_test'][:]
x = np.argmax(predict_val, axis = 1)
predict_valid = np.zeros(predict_val.shape)
predict_valid[x == 0,0] = 1
predict_valid[x == 1,1] = 1
predict_valid[x == 2,2] = 1

df = prediction_accuracy(ytest, predict_valid)

ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')

# now save the heatmap
ax.figure.savefig('figures/cnn_train_1000.png')
plt.clf()






