#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:40:27 2019

@author: TH
"""

from keras.models import load_model
import h5py
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

svhn_model = load_model('svhn.h5')
#%%
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

#%%

#y_ = np.array([1,2,3,4,5,6,7,8,9,0])
#enc = OneHotEncoder().fit(y_.reshape(-1, 1))
y_pred = svhn_model.predict_classes(X_test, batch_size=32, verbose=1)
#y_pred = enc.transform(y_pred.reshape(-1, 1)).toarray()
#%%

f1 = f1_score(y_test, y_pred, average = None)
output_string = ""
for i in range(0, len(f1)):
    output_string += "F1_score(target = {}): {}".format(i, f1[i])
    output_string += '\n'
#output_string = "target = {}: {}".format(0,f1[0])     
#output_string = "target = 0: {}".format(f1[0]) + "\n" + "target = 1: {}".format(f1[1]) + '\n' + "target = 3: {}".format(f1[3])
print (output_string)

#%%

from sklearn.metrics import confusion_matrix

# Set the figure size

#y_test.shape
#y_pred.shape
# Calculate the confusion matrix
cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred = y_pred)

# Normalize the confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

# Visualize the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True);
