#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:57:27 2019

@author: TH
"""

#print ("Descriptive Analysis")
import svhn
import numpy as np
import matplotlib.pyplot as plt
import h5py
svhn.downloadset()

h5f = h5py.File('SVHN_single_grey.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]

# Close this file
h5f.close()

X_trainori, y_trainori = load_data('train_32x32.mat')
X_testori, y_testori = load_data('test_32x32.mat')

X_trainori, y_trainori = X_trainori.transpose((3,0,1,2)), y_trainori[:,0]
X_testori, y_testori = X_testori.transpose((3,0,1,2)), y_testori[:,0]



def balanced_subsample(y, s):
    """Return a balanced subsample of the population"""
    sample = []
    # For every label in the dataset
    for label in np.unique(y):
        # Get the index of all images with a specific label
        images = np.where(y==label)[0]
        # Draw a random sample from the images
        random_sample = np.random.choice(images, size=s, replace=False)
        # Add the random sample to our subsample list
        sample += random_sample.tolist()
    return sample

y_trainori[y_trainori == 10] = 0
y_testori[y_testori == 10] = 0
    
train_samples = balanced_subsample(y_trainori, 600)
    
X_valori, y_valori = np.copy(X_trainori[train_samples]), np.copy(y_trainori[train_samples])

    # Remove the samples to avoid duplicates
X_trainori = np.delete(X_trainori, train_samples, axis=0)
y_trainori = np.delete(y_trainori, train_samples, axis=0)


plt.imshow(X_train[11][:,:,0], cmap='binary')
y_train[11]

plt.imshow(X_trainori[11])
X_trainori[3].shape
