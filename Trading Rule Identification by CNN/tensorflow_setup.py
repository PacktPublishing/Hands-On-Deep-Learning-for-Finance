import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
    segments, labels = segment_signal(df)
    list_ch_train = df.Action.unique()
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    X_tr, X_vld, lab_tr, lab_vld = train_test_split(segments, labels, stratify = labels, random_state = 123)

    return X_tr, X_vld, lab_tr, lab_vld, list_ch_train

def create_tensorflow_test_data(csvfilename):
    df = pd.read_csv('sampledata/'+ csvfilename)
    segments, labels = segment_signal(df)
    list_ch_test = df.Action.unique()
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    X_test = segments
    y_test = labels

    return X_test, y_test, list_ch_test






