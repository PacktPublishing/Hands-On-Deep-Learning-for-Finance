from tensorflow_setup import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os 
import matplotlib.pyplot as plt

# filenames used 
traindtfile = 'hdf_50.h5'
testdtfile = 'hdf_50.h5'
losssavefig = 'cnn_train_50_loss.png'
accsavefig = 'cnn_train_50_accuracy.png'
resultsave = 'cnn_train_50.h5'
chkpointdir = 'cnn-50/'

X_tr, X_vld, y_tr, y_vld, list_ch_train = get_tf_train_data(traindtfile)


batch_size = 600       # Batch size
seq_len = 12          # Number of steps
learning_rate = 0.0001
epochs = 100

n_classes = 3 # buy sell and nothing
n_channels = 6 # moving averages

graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

with graph.as_default():
    # (batch, 12, 3) --> (batch, 6, 6)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=6, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_1, (-1, 6*6))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    
    # Predictions
    logits = tf.layers.dense(flat, n_classes)
    
    soft = tf.argmax(logits,1)
    pred = tf.nn.softmax(logits,1)
    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


if (os.path.exists('bschkpnt-cnn') == False):
    os.system('mkdir bschkpnt-cnn')

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
   
    # Loop over epochs
    for e in range(epochs):
        
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
            
            # Loss
            loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 10 iterations
            if (iteration%10 == 0):                
                val_acc_ = []
                val_loss_ = []
                
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                    
                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            
            # Iterate 
            iteration += 1


    saver.save(sess,chkpointdir + "bs.ckpt")


t = np.arange(iteration-1)

plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('figures/'+losssavefig)

# change filename here


# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('figures/'+accsavefig)

del(X_tr, X_vld, y_tr, y_vld, list_ch_train)

X_test, y_test, lab_ch_test = get_tf_test_data(testdtfile)

test_acc = []
probs = []

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint(chkpointdir))
    
    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}
        
        batch_acc = sess.run(accuracy, feed_dict=feed)
        test_acc.append(batch_acc)
        prob = sess.run(pred, feed_dict=feed)
        probs.append(prob)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))


# now reshape the probs array
probs = np.array(probs)
probs = probs.reshape((probs.shape[0]*probs.shape[1]), probs.shape[2])
y_test = y_test[:len(probs),:]
# model complete

# save the results

import h5py
hf = h5py.File('h5files/' + resultsave, 'w')
hf.create_dataset('predict_valid', data=probs)
hf.create_dataset('y_test', data = y_test)
hf.close()

del(X_test, y_test, lab_ch_test)
