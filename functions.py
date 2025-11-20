"""
Various utility functions for data reporting, plotting, and saving results.
Author: Alexander Katrompas
"""


import config as cfg
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_actual_vs_predicted(actual, predicted):
    """
    Save actual vs predicted values to a CSV file.
    :param actual: numpy.ndarray, actual values
    :param predicted: numpy.ndarray, predicted values
    :return: None
    """
    ap_data = np.column_stack((actual, predicted))
    np.savetxt(cfg.APFILE, ap_data, delimiter=',', header='Actual,Predicted', comments='')
    print(f"Actual vs Predicted data saved to {cfg.APFILE}\n")


def data_report(df):
    """
    Generate a report of the dataframe including plots, info, and statistics.
    :param df: pandas.DataFrame
    :return: None
    """
    plot_features(df)
    print("df info:")
    print(df.info())
    print("\ndf statistics:")
    print(df.describe())
    print()


def nan_report(df):
    """
    Generate a report of NaN values in the dataframe.
    :param df: pandas.DataFrame
    :return: None
    """
    print("Generating NaN report...")
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    total_rows = len(df)
    print(f" * Rows in dataset: {total_rows}")
    print(f" * Total NaN values in dataset: {total_nans}")
    for col, count in nan_counts.items():
        if count > 0:
            print(f" * Column '{col}' has {count} NaN values: {count / total_rows:.{1}%}")
    print()


def reg_classification_report(actual, predicted, threshold):
    # create an array of actual absolute difference from predicted within threshold
    error = np.abs(actual - predicted)
    # get percent difference between actual and predicted
    percent_diff = error / np.abs(actual)

    # make array from percent difference within threshold
    # if percent difference <= threshold, set to 1, else 0
    predicted_cls = np.zeros(len(percent_diff))
    for i in range(len(percent_diff)):
        if percent_diff[i] <= cfg.CLASSIFICATION_THRESHOLD:
            predicted_cls[i] = 1
    actual_cls = np.ones_like(actual)

    acc = accuracy_score(actual_cls, predicted_cls)
    prec = precision_score(actual_cls, predicted_cls)
    rec = recall_score(actual_cls, predicted_cls)
    f1 = f1_score(actual_cls, predicted_cls)

    print("Classification Metrics (within {:.1%} error):".format(cfg.CLASSIFICATION_THRESHOLD))
    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(prec))
    print("Recall: {:.4f}".format(rec))
    print("F1 Score: {:.4f}".format(f1))


def plot_actual_vs_predicted(actual, predicted):
    """
    Plot actual vs predicted values.
    :param actual: numpy.ndarray
    :param predicted: numpy.ndarray
    :return: None
    """
    pyplot.figure()
    pyplot.plot(actual, label='Actual')
    pyplot.plot(predicted, label='Predicted')
    pyplot.title('Actual vs Predicted Pollution Levels')
    pyplot.xlabel('Time Step')
    pyplot.ylabel('Normalized Pollution Level')
    pyplot.legend()
    pyplot.show()


def plot_training(history):
    """
    Plot training and validation loss over epochs.
    :param history: keras.callbacks.History
    :return:
    """
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


def plot_features(df):
    """
    Plot all features in the dataframe.
    :param df: pandas.DataFrame
    :return: None
    """
    values = df.values
    # specify columns to plot, skip categorical variable 'wnd_dir'
    groups = [0, 1, 2, 3, 4, 5, 9]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(df.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()
