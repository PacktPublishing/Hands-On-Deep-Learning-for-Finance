import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(data,window_size = 12):
    segments = np.empty((0,window_size,6))
    labels = np.empty((0))
    for (start, end) in windows(data['Date'], window_size):
        x = data["mav5"][start:end]
        y = data["mav10"][start:end]
        z = data["mav20"][start:end]
        a = data["mav30"][start:end]
        b = data["mav50"][start:end]
        c = data["mav100"][start:end]
        if(len(data['Date'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x,y,z,a,b,c])])
            labels = np.append(labels,stats.mode(data["Action"][start:end])[0][0])
    return segments, labels


def get_batches(X, y, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]


def create_tensorflow_train_data(csvfilename):
    df = pd.read_csv('sampledata/'+ csvfilename)
    df = df[['Date','symbolid','buyret','sellret','Action','mav5', 'mav10','mav20','mav30','mav50','mav100']]
    symbols = df.symbolid.unique()
    segments, labels = segment_signal(df[df.symbolid == symbols[0]])
    df = df[df.symbolid != symbols[0]]
    symbols = symbols[1:]
    for i in range(0,len(symbols)):
        x, a = segment_signal(df[df.symbolid == symbols[i]])
        segments = np.concatenate((segments, x), axis = 0)
        labels = np.concatenate((labels, a), axis = 0)
        df = df[df.symbolid != symbols[i]]
        print(str(round(i/len(symbols)*100,2)) + ' percent done')
    list_ch_train = pd.get_dummies(labels)
    list_ch_train = np.asarray(list_ch_train.columns)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    X_tr, X_vld, lab_tr, lab_vld = train_test_split(segments, labels, stratify = labels, random_state = 123)

    return X_tr, X_vld, lab_tr, lab_vld, list_ch_train

def create_tensorflow_test_data(csvfilename):
    df = pd.read_csv('sampledata/'+ csvfilename)
    df = df[['Date','symbolid','buyret','sellret','Action','mav5', 'mav10','mav20','mav30','mav50','mav100']]
    list_ch_test = df.Action.unique()
    symbols = df.symbolid.unique()
    segments, labels = segment_signal(df[df.symbolid == symbols[0]])
    df = df[df.symbolid != symbols[0]]
    symbols = symbols[1:]
    for i in range(0,len(symbols)):
        x, a = segment_signal(df[df.symbolid == symbols[i]])
        segments = np.concatenate((segments, x), axis = 0)
        labels = np.concatenate((labels, a), axis = 0)
        df = df[df.symbolid != symbols[i]]
        print(str(round(i/len(symbols)*100,2)) + ' percent done')

    list_ch_test = pd.get_dummies(labels)
    list_ch_test = np.asarray(list_ch_test.columns)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    X_test = segments
    y_test = labels

    return X_test, y_test, list_ch_test


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

# compute the type 1 and type 2 errors
def prediction_accuracy(ytest, predict_val):

    # rows belong to predicton
    # columns to test values
    # order BUY, NONE , SELL
    accuracy_mat = np.zeros([3,3], dtype = float)
    for i in range(ytest.shape[1]):
        for j in range(predict_val.shape[1]):
            accuracy_mat[i,j] = sum(predict_val[(predict_val[:,j] *  ytest[:,i] > 0),j])
    allobs = sum(map(sum, accuracy_mat))
    accuracy_mat = np.divide(accuracy_mat, allobs)*100
    accuracy_mat = pd.DataFrame(accuracy_mat, columns = ['Buy', 'None', 'Sell'], index = ['Buy', 'None', 'Sell'])
    return accuracy_mat



