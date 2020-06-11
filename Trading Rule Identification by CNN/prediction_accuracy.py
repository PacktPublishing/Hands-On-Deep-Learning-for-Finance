import numpy as np
import pandas as pd

# compute the accuracy of predictions given test values and predicted values

def prediction_accuracy(ytest, predict_val):
 rows belong to predicton
 # columns to test values
 # order BUY, NONE , SELL
 accuracy_mat = np.zeros([3,3], dtype = float)
 for i in range(ytest.shape[1]):
 for j in range(predict_val.shape[1]):
     accuracy_mat[i,j] = sum(predict_val[(predict_val[:,j] * ytest[:,i] > 0),j])
 allobs = sum(map(sum, accuracy_mat))
 accuracy_mat = np.divide(accuracy_mat, allobs)*100
 accuracy_mat = pd.DataFrame(accuracy_mat, columns = ['Buy', 'None', 'Sell'], index = ['Buy', 'None', 'Sell'])
 return accuracy_mat
