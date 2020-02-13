# simple classification using neural network
# adapted from iris classification
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image
from pandas import get_dummies
from sklearn.cross_validation import train_test_split
# Config the matlotlib backend as plotting inline in IPython
%matplotlib inline

# read the sample data file

csvfilename = 'train_50.csv'

data = pd.read_csv('sampledata/'+ csvfilename)

# reshape dataframe
data = data[['mav5', 'mav10', 'mav20', 'mav30', 'mav50', 'mav100', 'Action']]
# plot to visualise the data
g=sns.pairplot(data, hue="Action", height= 2.5)
g.savefig('figures/train_50_desc.png')

cols = data.columns
features = cols[0:6]
labels = cols[6]
print(features)
print(labels)

#Shuffle The data
X = data[features]
y = data[labels]

y = get_dummies(y)

X_train = np.array(X).astype(np.float32)
y_train = np.array(y).astype(np.float32)


csvfilename = 'test_50.csv'
data = pd.read_csv('sampledata/'+ csvfilename)

# reshape dataframe
data = data[['mav5', 'mav10', 'mav20', 'mav30', 'mav50', 'mav100', 'Action']]

g=sns.pairplot(data, hue="Action", height= 2.5)
g.savefig('figures/test_50_desc.png')

X = data[features]
y = data[labels]

y = get_dummies(y)

X_test  = np.array(X).astype(np.float32)
y_test  = np.array(y).astype(np.float32)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

training_size = X_train.shape[1]
test_size = X_test.shape[1]
num_features = 6
num_labels = 3


num_hidden = 10

graph = tf.Graph()
with graph.as_default():
    tf_train_set    = tf.constant(X_train)
    tf_train_labels = tf.constant(y_train)
    tf_valid_set    = tf.constant(X_test)
 
    
    print(tf_train_set)
    print(tf_train_labels)
    
    ## Note, since there is only 1 layer there are actually no hidden layers... but if there were
    ## there would be num_hidden
    weights_1 = tf.Variable(tf.truncated_normal([num_features, num_hidden]))
    weights_2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    ## tf.zeros Automaticaly adjusts rows to input data batch size
    bias_1 = tf.Variable(tf.zeros([num_hidden]))
    bias_2 = tf.Variable(tf.zeros([num_labels]))
    
    
    logits_1 = tf.matmul(tf_train_set , weights_1 ) + bias_1
    rel_1 = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(rel_1, weights_2) + bias_2
    
    soft = tf.nn.softmax(logits_2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(.005).minimize(loss)
    
    
    ## Training prediction
    predict_train = tf.nn.softmax(logits_2)
    
        
    # Validation prediction
    logits_1_val = tf.matmul(tf_valid_set, weights_1) + bias_1
    rel_1_val    = tf.nn.relu(logits_1_val)
    logits_2_val = tf.matmul(rel_1_val, weights_2) + bias_2
    predict_valid = tf.nn.softmax(logits_2_val)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with graph.as_default():
    saver = tf.train.Saver()

if (os.path.exists('simpleclass') == False):
    !mkdir bschkpnt-cnn

num_steps = 10000
with tf.Session(graph = graph) as session:
    session.run(tf.global_variables_initializer())
    print(loss.eval())
    for step in range(num_steps):
        _,l, predictions = session.run([optimizer, loss, predict_train])
        
        if (step % 2000 == 0):
              #print(predictions[3:6])
              print('Loss at step %d: %f' % (step, l))
              print('Training accuracy: %.1f%%' % accuracy( predictions, y_train[:, :]))
              print('Validation accuracy: %.1f%%' % accuracy(predict_valid.eval(), y_test))
              predict_valid_arr = predict_valid.eval()
              
    saver.save(session,"simpleclass/simple.ckpt")

# save the results

import h5py
hf = h5py.File('h5files/simpleclass_train_50.h5', 'w')
hf.create_dataset('predict_valid', data=predict_valid_arr)
hf.create_dataset('y_test', data = y_test)
hf.close()


