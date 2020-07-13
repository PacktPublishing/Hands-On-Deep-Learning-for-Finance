These set of files are used for modeling trading rules using moving averages

datasetup - jupyter notebook to get data from yahoo finance and save it as csv files in csvfiles directory for later use.
            symbollist provided in descdata.csv file in csvfiles directory to download the data for nasdaq symbols

trading rule - jupyter notebook to determine the efficacy of a simple moving average trading rule and assess confusion matrix for its efficacy

simple classifier tf1 - jupyter notebook implementing a simple classifier inspired by iris classification problem to check increased accuracy relative to simple trading rule

tensorflow setup - set of functions used for running a CNN classifier

savedatasets - create datasets from csv files for a random number of assets and save them as h5 files for further use in tensorflow execution

tensorflow_execute_XX.py - train and test a CNN model for different data sizes. The relevance of doing it for multiple data sizes is to assess the deterioration in performance once sample size is increased. We are dealing with an unbalanced dataset.
