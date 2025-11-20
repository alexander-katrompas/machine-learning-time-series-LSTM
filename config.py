"""
Configuration settings for the PPM prediction model.
Author: Alexander Katrompas
"""

VERBOSE = True
LOGLEVEL = '3'
RAWDATAFILE = "ppm_raw.csv"
PROCESSEDDATAFILE = "ppm_cleaned.csv"
NORMALIZEDDATAFILE = "ppm_normalized.csv"
APFILE = "ppm_actual_predicted.csv"

SEQUENCE_LENGTH = 3
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10

LSTM_UNITS = 10
DENSE1_UNITS = 5
DENSE2_UNITS = 2

CLASSIFICATION_THRESHOLD = 0.30  # error threshold for classification metrics
