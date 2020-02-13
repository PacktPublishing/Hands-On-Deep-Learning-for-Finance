#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, shutil
import random
import gc  #Gabage collector for cleaning deleted data from memory

import cv2
import numpy as np

#Lets declare our image dimensions
#we are using coloured images. 
nrows = 32
ncolumns = 32
channels = 3  #change to 1 if you want to use grayscale image


#A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = [] # labels
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
        #get the labels
        if 'Buy' in image:
            y.append(1)
        elif 'Sell' in image:
            y.append(2)
        else: # None
            y.append(3)
    
    return X, y



def get_data():

    train_dir = 'input/train'
    test_dir = 'input/test'

    # train_imgs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir)]  #get full data set
    train_buys = ['input/train/{}'.format(i) for i in os.listdir(train_dir) if 'Buy' in i]  #get dog images
    train_sells = ['input/train/{}'.format(i) for i in os.listdir(train_dir) if 'Sell' in i]  #get cat images
    train_none = ['input/train/{}'.format(i) for i in os.listdir(train_dir) if 'None' in i]  #get cat images

    test_imgs = ['input/test/{}'.format(i) for i in os.listdir(test_dir)] #get test images

    train_imgs = train_buys[:2000] + train_sells[:2000] + train_none[:2000]  # slice the dataset and use 2000 in each class
    random.shuffle(train_imgs)  # shuffle it randomly

    #Clear list that are useless
    del train_buys
    del train_sells
    del train_none
    gc.collect()

    #get the train and label data
    X, y = read_and_process_image(train_imgs)

    del train_imgs
    gc.collect()
    #Convert list to numpy array
    X = np.array(X)
    y = np.array(y)

    #Lets split the data into train and test set
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=2)

    X, y = read_and_process_image(test_imgs[:100])
    X_test = np.array(X)
    y_test = np.array(y)

    #clear memory
    del X
    del y
    del test_imgs
    gc.collect()


    return X_train, y_train, X_valid, y_valid, X_test, y_test