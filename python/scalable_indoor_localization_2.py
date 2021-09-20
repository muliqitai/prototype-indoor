#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     scalable_indoor_localization.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2017-11-15
#           2020-12-20 (updated for TensorFlow ver. 2.x)
#
# @brief    Build and evaluate a scalable indoor localization system
#           based on Wi-Fi fingerprinting using a neural-network-based
#           multi-label classifier.
#
# @remarks  This work is based on the <a href="https://keras.io/">Keras</a>-based
#           implementation of the system described in "<a
#           href="https://arxiv.org/abs/1611.02049v2">Low-effort place
#           recognition with WiFi fingerprints using deep learning</a>".
#
#           The results are published in the following paper:
#           Kyeong Soo Kim, Sanghyuk Lee, and Kaizhu Huang "A scalable deep
#           neural network architecture for multi-building and multi-floor
#           indoor localization based on Wi-Fi fingerprinting," Big Data
#           Analytics, vol. 3, no. 4, pp. 1-17, Apr. 19, 2018. Available online:
#           https://doi.org/10.1186/s41044-018-0031-2
#


### import modules (except tensorflow/keras)
import argparse
import datetime
import os
import math
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from timeit import default_timer as timer
from keras.models import load_model
import os.path as osp
from keras import backend as K

### global constant variables
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
INPUT_DIM = 520                 #  number of APs
VERBOSE = 1                     # 0 for turning off logging
#------------------------------------------------------------------------
# stacked auto encoder (sae)
#------------------------------------------------------------------------
# SAE_ACTIVATION = 'tanh'
SAE_ACTIVATION = 'relu'
SAE_BIAS = False
SAE_OPTIMIZER = 'adam'
SAE_LOSS = 'mse'
#------------------------------------------------------------------------
# classifier
#------------------------------------------------------------------------
CLASSIFIER_ACTIVATION = 'relu'
CLASSIFIER_BIAS = False
CLASSIFIER_OPTIMIZER = 'adam'
CLASSIFIER_LOSS = 'binary_crossentropy'
#------------------------------------------------------------------------
# input files
#------------------------------------------------------------------------
path_train = '../data/UJIIndoorLoc/trainingData2.csv'           # '-110' for the lack of AP.
path_validation = '../data/UJIIndoorLoc/validationData2.csv'    # ditto
#------------------------------------------------------------------------
# output files
#------------------------------------------------------------------------
path_base = '../results/' + os.path.splitext(os.path.basename(__file__))[0]
path_out =  path_base + '_out'
path_sae_model = path_base + '_sae_model.hdf5'

#input_path = 'input path'
#weight_file = 'model.h5'
#weight_file_path = osp.join(input_path,weight_file)
#output_graph_name = weight_file[:-3] + '.pb'

def dnn_model():

    path_out = path_base + '_out'
    args = parser.parse_args()
# set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    training_ratio = args.training_ratio
    sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.classifier_hidden_layers == '':
        classifier_hidden_layers = ''
    else:
        classifier_hidden_layers = [int(i) for i in (args.classifier_hidden_layers).split(',')]
    dropout = args.dropout
    N = args.neighbours
    scaling = args.scaling

    ### initialize random seed generator of numpy
    np.random.seed(random_seed)
    
    #--------------------------------------------------------------------
    # import tensorflow/keras
    #--------------------------------------------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    import tensorflow as tf
    tf.random.set_seed(random_seed)  # initialize random seed generator of tensorflow
    from tensorflow import keras
    import keras
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import Sequential
    from tensorflow.keras.models import load_model

    # read both train and test dataframes for consistent label formation through one-hot encoding
    train_df = pd.read_csv(path_train, header=0)  # pass header=0 to be able to replace existing names
    test_df = pd.read_csv(path_validation, header=0)
    
    train_AP_features = scale(np.asarray(train_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
    train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1) # add a new column
    
    blds = np.unique(train_df[['BUILDINGID']])
    flrs = np.unique(train_df[['FLOOR']])
    x_avg = {}
    y_avg = {}
    for bld in blds:
        for flr in flrs:
            # map reference points to sequential IDs per building-floor before building labels
            cond = (train_df['BUILDINGID']==bld) & (train_df['FLOOR']==flr)
            _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True) # refer to numpy.unique manual
            train_df.loc[cond, 'REFPOINT'] = idx
            
            # calculate the average coordinates of each building/floor
            x_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LONGITUDE'])
            y_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LATITUDE'])
    
    # build labels for multi-label classification
    len_train = len(train_df)
    blds_all = np.asarray(pd.get_dummies(pd.concat([train_df['BUILDINGID'], test_df['BUILDINGID']]))) # for consistency in one-hot encoding for both dataframes
    flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'], test_df['FLOOR']]))) # ditto
    blds = blds_all[:len_train]
    flrs = flrs_all[:len_train]
    rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
    train_labels = np.concatenate((blds, flrs, rfps), axis=1)
    # labels is an array of 19937 x 118
    # - 3 for BUILDINGID
    # - 5 for FLOOR,
    # - 110 for REFPOINT
    OUTPUT_DIM = train_labels.shape[1]
    
    # split the training set into training and validation sets; we will use the
    # validation set at a testing set.
    train_val_split = np.random.rand(len(train_AP_features)) < training_ratio # mask index array
    x_train = train_AP_features[train_val_split]
    y_train = train_labels[train_val_split]
    x_val = train_AP_features[~train_val_split]
    y_val = train_labels[~train_val_split]

    ### build SAE encoder model
    print("\nPart 1: buidling an SAE encoder ...")
    # if False:
    if os.path.isfile(path_sae_model) and (os.path.getmtime(path_sae_model) > os.path.getmtime(__file__)):
        model = load_model(path_sae_model)
    else:
        # create a model based on stacked autoencoder (SAE)
        model = Sequential()
        model.add(Dense(sae_hidden_layers[0], name="sae-hidden-0", input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        for i in range(1, len(sae_hidden_layers)):
            model.add(Dense(sae_hidden_layers[i], name="sae-hidden-"+str(i), activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        model.add(Dense(INPUT_DIM, name="sae-output", activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)

        # train the model
        model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

        # remove the decoder part
        num_to_remove = (len(sae_hidden_layers) + 1) // 2
        for i in range(num_to_remove):
            model.pop()

        # # set all layers (i.e., SAE encoder) to non-trainable (weights will not be updated)
        # for layer in model.layers[:]:
        #     layer.trainable = False
        
        # save the model for later use
        #model.save(path_sae_model)

    ### build and train a complete model with the trained SAE encoder and a new classifier
    print("\nPart 2: buidling a complete model ...")
    # append a classifier to the model
    # class_weight = {
    #     0: building_weight, 1: building_weight, 2: building_weight,  # buildings
    #     3: floor_weight, 4: floor_weight, 5: floor_weight, 6:floor_weight, 7: floor_weight  # floors
    # }
    model.add(Dropout(dropout))
    for i in range(len(classifier_hidden_layers)):
        model.add(Dense(classifier_hidden_layers[i], name="classifier-hidden"+str(i), activation=CLASSIFIER_ACTIVATION, use_bias=CLASSIFIER_BIAS))
        model.add(Dropout(dropout))
    model.add(Dense(OUTPUT_DIM, name="activation-0", activation='sigmoid', use_bias=CLASSIFIER_BIAS))  # 'sigmoid' for multi-label classification
    model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])

    # train the model
    startTime = timer()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=VERBOSE)
    elapsedTime = timer() - startTime
    print("Model trained in %e s." % elapsedTime)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help="ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="random seed",
        default=0,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 20",
        default=20,
        type=int)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 10",
        default=10,
        type=int)
    parser.add_argument(
        "-T",
        "--training_ratio",
        help="ratio of training data to overall data: default is 0.9",
        default=0.9,
        type=float)
    parser.add_argument(
        "-S",
        "--sae_hidden_layers",
        help=
        "comma-separated numbers of units in SAE hidden layers; default is '256,128,64,128,256'",
        default='256,128,64,128,256',
        type=str)
    parser.add_argument(
        "-C",
        "--classifier_hidden_layers",
        help=
        "comma-separated numbers of units in classifier hidden layers; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "-D",
        "--dropout",
        help=
        "dropout rate before and after classifier hidden layers; default 0.0",
        default=0.0,
        type=float)
    parser.add_argument(
        "-N",
        "--neighbours",
        help="number of (nearest) neighbour locations to consider in positioning; default is 1",
        default=1,
        type=int)
    parser.add_argument(
        "--scaling",
        help=
        "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.0",
        default=0.0,
        type=float)
    dnn_model()


    #model.save(export_path_keras)
    #loaded = keras.models.load_model("my_model")


    #h5_model = load_model(weight_file_path)
    #dnn_model(h5_model, model_name=output_graph_name)
