from scalable_indoor_localization import dnn_model
import argparse
import datetime
import os
import math
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from timeit import default_timer as timer
import keras
# load the model
model=keras.models.load_model("my_model")
batch_size=10

path_train = '../data/UJIIndoorLoc/trainingData2.csv'
path_validation = '../data/UJIIndoorLoc/validationData2.csv'
test_df = pd.read_csv(path_validation, header=0)
# turn the given validation set into a testing set

train_df = pd.read_csv(path_train, header=0)  # pass header=0 to be able to replace existing names


train_AP_features = scale(np.asarray(train_df.iloc[:, 0:520]).astype(float),
                          axis=1)  # convert integer to float and scale jointly (axis=1)
train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1)

# turn the given validation set into a testing set
test_AP_features = scale(np.asarray(test_df.iloc[:, 0:520]).astype(float),
                         axis=1)  # convert integer to float and scale jointly (axis=1)
x_test_utm = np.asarray(test_df['LONGITUDE'])
y_test_utm = np.asarray(test_df['LATITUDE'])
blds_all = np.asarray(pd.get_dummies(pd.concat([train_df['BUILDINGID'], test_df['BUILDINGID']]))) # for consistency in one-hot encoding for both dataframes
flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'], test_df['FLOOR']])))
len_train = len(train_df)
blds = blds_all[len_train:]
flrs = flrs_all[len_train:]

### evaluate the model
print("\nPart 3: evaluating the model ...")

# calculate the building and floor estimation
preds = model.predict(test_AP_features, batch_size=batch_size)  #information about the buidling...
blds_results = (np.equal(np.argmax(blds, axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
flrs_results = (np.equal(np.argmax(flrs, axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
mask = np.logical_and(blds_results, flrs_results)
rfps = (preds[mask])[:, 8:118]
print('preds',preds)
print('building',np.argmax(preds[:, :3]))
print('floor',flrs_results)
print('mask',preds[mask])
print('rfps',rfps)

