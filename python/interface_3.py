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
N=1
scaling=0.0
#user part

path_train = '../data/UJIIndoorLoc/trainingData2.csv'
path_validation = '../data/UJIIndoorLoc/validationData2.csv'
#a = st.file_uploader("Choose a CSV file", type="csv")

#if a is not None:
    #path_validation = a
test_df = pd.read_csv(path_validation, header=0)
# turn the given validation set into a testing set

train_df = pd.read_csv(path_train, header=0)  # pass header=0 to be able to replace existing names

test=test_df.sample(n=1)
train=train_df.sample(n=1)

train_AP_features = scale(np.asarray(train_df.iloc[:,0:520]).astype(float),axis=1)  # convert integer to float and scale jointly (axis=1)
train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1)# add a new column
blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])
x_avg = {}
y_avg = {}
for bld in blds:
    for flr in flrs:
        # map reference points to sequential IDs per building-floor before building labels
        cond = (train_df['BUILDINGID'] == bld) & (train_df['FLOOR'] == flr)
        _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True)  # refer to numpy.unique manual
        train_df.loc[cond, 'REFPOINT'] = idx

        # calculate the average coordinates of each building/floor
        x_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LONGITUDE'])
        y_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LATITUDE'])

test_AP_features = scale(np.asarray(test.iloc[:,0:520]).astype(float),axis=1)  # convert integer to float and scale jointly (axis=1)
blds_all = np.asarray(pd.get_dummies(pd.concat([train_df['BUILDINGID'], test_df['BUILDINGID']]))) # for consistency in one-hot encoding for both dataframes
flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'], test_df['FLOOR']])))
len_train = len(train_df)

blds = blds_all[:len_train]
flrs = flrs_all[:len_train]
rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
train_labels = np.concatenate((blds, flrs, rfps), axis=1)

# turn the given validation set into a testing set
 # convert integer to float and scale jointly (axis=1)
blds_all = np.asarray(pd.get_dummies(pd.concat([train['BUILDINGID'], test['BUILDINGID']]))) # for consistency in one-hot encoding for both dataframes
flrs_all = np.asarray(pd.get_dummies(pd.concat([train['FLOOR'], test['FLOOR']])))
len_train = len(train)
x_test_utm = np.asarray(test['LONGITUDE'])
y_test_utm = np.asarray(test['LATITUDE'])
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
print('building',blds_results)
print('floor',flrs_results)
print('mask',mask)
if mask == [False]:
    print('The prediction result is False')
#print('rfps',preds[0:1, 8:118])
#print('rfps',rfps)

st.write('preds',preds)
st.write('building',blds_results)
st.write('floor',flrs_results)
#st.write('rfps',rfps)
#print('blds',preds[:, :3])
#print('flrs',preds[:, 3:8])
#rfps = (preds[mask])[:, 8:118]
#print('rfps',rfps)

x_test_utm = x_test_utm[mask]
y_test_utm = y_test_utm[mask]
blds = blds[mask]
flrs = flrs[mask]
rfps = (preds[mask])[0:1, 8:118]
n_success = len(blds)  # number of correct building and floor location
n_loc_failure = 0
sum_pos_err = 0.0
sum_pos_err_weighted = 0.0
idxs = np.argpartition(rfps, -N)[:, -N:]  # (unsorted) indexes of up to N nearest neighbors
threshold = scaling*np.amax(rfps, axis=1)
Cor = [[] for _ in range(n_success)]
for i in range(n_success):
    xs = []
    ys = []
    ws = []
    for j in idxs[i]:
        rfp = np.zeros(110)
        rfp[j] = 1
        rows = np.where((train_labels == np.concatenate((blds[i], flrs[i], rfp))).all(axis=1)) # tuple of row indexes
        if rows[0].size > 0:
            if rfps[i][j] >= threshold[i]:
                xs.append(train.loc[train.index[rows[0][0]], 'LONGITUDE'])
                ys.append(train.loc[train.index[rows[0][0]], 'LATITUDE'])
                ws.append(rfps[i][j])
    if len(xs) > 0:
        sum_pos_err += math.sqrt((np.mean(xs) - x_test_utm[i]) ** 2 + (np.mean(ys) - y_test_utm[i]) ** 2)
        sum_pos_err_weighted += math.sqrt((np.average(xs, weights=ws) - x_test_utm[i]) ** 2 + (np.average(ys, weights=ws) - y_test_utm[i]) ** 2)
        Cor[i].append(np.average(xs, weights=ws))
        Cor[i].append(np.average(ys, weights=ws))
        #print('x_1,y_1',np.average(xs, weights=ws),np.average(ys, weights=ws))
    else:
        n_loc_failure += 1
        key = str(np.argmax(blds[i])) + '-' + str(np.argmax(flrs[i]))
if mask == [True]:
    print('Cor',Cor)




