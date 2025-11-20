"""
This module contains functions to create and evaluate an LSTM model for time series prediction.
Author: Alexander Katrompas
"""

import config as cfg
import functions as fn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

##############################################
# the following is to make TF quiet (optional)
##############################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
from silenceStdError import SilenceStdErr as silence
import silence_tensorflow.auto
import absl.logging
absl.logging.use_absl_handler()
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
###############################################


def evaluate_model(model, testX, testY, save_actual_predicted=False, reg_class_report=False):
    """
    Evaluate the model on the test set and return the MAE.
    :param model: keras.Sequential model
    :param testX: numpy array, test input data
    :param testY: numpy array, test target data
    :return: float, MAE, MSE, R^2 on test set
    :note: confusion matrix and classification report are not typically used for
           regression tasks like this, but we can still calculate regression metrics.
    """
    with silence():
        model.evaluate(testX, testY, verbose=0)
    predY = model.predict(testX).flatten()
    fn.plot_actual_vs_predicted(testY, predY)

    if save_actual_predicted:
        fn.save_actual_vs_predicted(testY, predY)

    mae = mean_absolute_error(testY, predY)
    mse = mean_squared_error(testY, predY)
    r2 = r2_score(testY, predY)
    print("Test MAE: {:.4f}".format(mae))
    print("Test MSE: {:.4f}".format(mse))
    print("Test R^2: {:.4f}".format(r2))

    # although this is a regression task, we can compute classification metrics
    # this is almost never done academically, but can be useful in practice
    # especially for mission-critical applications where being within a certain error margin
    # is required. Example: prediction considered correct if error ≤ 5%
    if reg_class_report:
        fn.reg_classification_report(testY, predY, cfg.CLASSIFICATION_THRESHOLD)

    return mae, mse, r2


def create_lstm_model(input_shape):
    """
    Create and return an LSTM model for time series prediction.
    :param input_shape: tuple, shape of the input data (timesteps, features)
    :return: keras.Sequential model
    """
    with silence():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))  # first layer: input definition
        model.add(tf.keras.layers.LSTM(cfg.LSTM_UNITS))  # second layer: LSTM
        model.add(tf.keras.layers.Dense(cfg.DENSE1_UNITS))
        model.add(tf.keras.layers.Dropout(0.2))  # dropout for regularization
        model.add(tf.keras.layers.Dense(1))  # output layer
        model.compile(loss='mae', optimizer='adam')

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.PATIENCE,
            restore_best_weights=True
        )

    return model, early_stop
