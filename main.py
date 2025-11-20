#!.venv/bin/python3
"""
Main script to load, process, train, and evaluate the LSTM model for pollution prediction.
Assumes the presence of the following modules:
- dataprocessing.py: for loading and processing the dataset
- config.py: for configuration parameters
- functions.py: for utility functions like data reporting and plotting
- model.py: for creating and evaluating the LSTM model
Author: Alexander Katrompas
"""

import dataprocessing as dp
import config as cfg
import functions as fn
import model as mdl

import numpy as np
from sklearn.preprocessing import MinMaxScaler


if cfg.VERBOSE: print("Loading and processing dataset.\n")
dataset = dp.load_clean_save(True)
if cfg.VERBOSE: print("Loading and processing dataset done.\n")

if cfg.VERBOSE: fn.data_report(dataset)

# normalize features
if cfg.VERBOSE: print("Normalizing dataset...", end="")
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(dataset.values)
del scaler, dataset # we don't need these anymore
if cfg.VERBOSE: print("done.\n")

# At this point, 'data' contains the normalized dataset ready for further processing.
# save data to csv for reference (directly from numpy array)
np.savetxt(cfg.NORMALIZEDDATAFILE, data, delimiter=',')

# this is a time series problem, so we need to convert the data into sequences
# we will use a sliding window approach to create sequences
# this will create sequences of length cfg.SEQUENCE_LENGTH
# assuming the last column is the target variable (pollution)
# and we are going to predict 1-step future pollution based on previous SEQUENCE_LENGTH time steps
if cfg.VERBOSE: print("Creating sequences from dataset...", end="")
data = dp.create_sequences(data, cfg.SEQUENCE_LENGTH)
if cfg.VERBOSE: print("done.\n")

# uncomment these to check against the cfg.NORMALIZEDDATAFILE
#print(data[0])
#print(data[0].shape)
#print()
#print(data[1])
#print(data[1].shape)
#print()

# split data into train, validation, and test sets
# since this is time series data, we will not shuffle the data
total_size = len(data[0])
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.15)
test_size = total_size - train_size - val_size

# will use the first 70% for training, next 15% for validation, last 15% for testing
if cfg.VERBOSE: print("Splitting data into train, validation, and test sets...", end="")
trainX, trainY = data[0][:train_size], data[1][:train_size]
valX, valY = data[0][train_size:train_size+val_size], data[1][train_size:train_size+val_size]
testX, testY = data[0][train_size+val_size:], data[1][train_size+val_size:]
if cfg.VERBOSE: print("done.\n")

if cfg.VERBOSE: print("Data preparation complete. Ready for model training/evaluation.\n")

# create LSTM network architecture and train the model
model, stop = mdl.create_lstm_model((cfg.SEQUENCE_LENGTH, trainX.shape[2]))
model.summary()
history = model.fit(trainX, trainY,
                    epochs=cfg.EPOCHS,
                    batch_size=cfg.BATCH_SIZE,
                    validation_data=(valX, valY),
                    verbose=2, callbacks=[stop])
fn.plot_training(history)
mdl.evaluate_model(model, testX, testY, True, True)
