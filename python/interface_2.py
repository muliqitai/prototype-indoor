import argparse
import datetime
import os
import math
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from timeit import default_timer as timer
from tensorflow import keras
import streamlit as st

st.title('A wifi indoor localization web app')
# load the model
model=keras.models.load_model("my_model")
batch_size=10
#user part

path_train = '../data/UJIIndoorLoc/trainingData2.csv'
#path_validation = '../data/UJIIndoorLoc/validationData2.csv'
a = st.file_uploader("Choose a CSV file", type="csv")

if a is not None:
    path_validation = a
test_df = pd.read_csv(path_validation, header=0)
# turn the given validation set into a testing set

train_df = pd.read_csv(path_train, header=0)  # pass header=0 to be able to replace existing names


train_AP_features = scale(np.asarray(train_df.iloc[0:1, 0:520]).astype(float),
                          axis=1)  # convert integer to float and scale jointly (axis=1)
train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1)

# turn the given validation set into a testing set
test_AP_features = scale(np.asarray(test_df.iloc[0:1, 0:520]).astype(float),
                         axis=1)  # convert integer to float and scale jointly (axis=1)
x_test_utm = np.asarray(test_df['LONGITUDE'].iloc[0:1])
y_test_utm = np.asarray(test_df['LATITUDE'].iloc[0:1])
blds_all = np.asarray(pd.get_dummies(pd.concat([train_df['BUILDINGID'].iloc[0:1], test_df['BUILDINGID'].iloc[0:1]]))) # for consistency in one-hot encoding for both dataframes
flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'].iloc[0:1], test_df['FLOOR'].iloc[0:1]])))
len_train = len(train_df.iloc[0:1])
blds = blds_all[len_train:]
flrs = flrs_all[len_train:]

### evaluate the model
print("\nPart 3: evaluating the model ...")

# calculate the building and floor estimation
preds = model.predict(test_AP_features, batch_size=batch_size)  #information about the buidling...
blds_results = (np.equal(np.argmax(blds, axis=1), np.argmax(preds[0:1, :3], axis=1))).astype(int)
flrs_results = (np.equal(np.argmax(flrs, axis=1), np.argmax(preds[0:1, 3:8], axis=1))).astype(int)
mask = np.logical_and(blds_results, flrs_results)
rfps = (preds[mask])[:, 8:118]
print('preds',preds)
print('building',blds_results)
print('floor',flrs_results)
#print('rfps',preds[0:1, 8:118])
print('rfps',rfps)

st.write('preds',preds)
st.write('building',blds_results)
st.write('floor',flrs_results)
st.write('rfps',rfps)
#print('blds',preds[:, :3])
#print('flrs',preds[:, 3:8])
#rfps = (preds[mask])[:, 8:118]
#print('rfps',rfps)







