from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def log_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1. - eps)
    ce = -(np.sum(y_true * np.log(y_pred), axis=1))
    mce = ce.mean()
    return mce


def call_accuracies(confusion_val):
    accuracies = []
    for i in range(confusion_val.shape[0]):
        num = confusion_val[i, :].sum()
        if num:
            accuracies.append(confusion_val[i, i] / num)
        else:
            accuracies.append(0.0)
    return np.float32(accuracies)


def accuracy(confusion_val):
    num_correct = 0
    for i in range(confusion_val.shape[0]):
        num_correct += confusion_val[i, i]
    return float(num_correct) / confusion_val.sum()


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_data, validation_steps, wanted_words, all_words,
                 label2int):
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.wanted_words = wanted_words
        self.all_words = all_words
        self.label2int = label2int
        self.int2label = {v: k for k, v in label2int.items()}

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for i in range(self.validation_steps):
            x_batch, y_true_batch = next(self.validation_data)
            y_pred_batch = self.model.predict(x_batch)

            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)

        y_true = np.float32(y_true)
        y_pred = np.float32(y_pred)
        val_loss = log_loss(y_true, y_pred)
        # map integer labels to strings
        y_true = list(y_true.argmax(axis=-1))
        y_pred = list(y_pred.argmax(axis=-1))
        y_true = [self.int2label[y] for y in y_true]
        y_pred = [self.int2label[y] for y in y_pred]
        confusion = confusion_matrix(y_true, y_pred)
        accs = call_accuracies(confusion)
        acc = accuracy(confusion)
        # same for wanted words
        y_true = [y if y in self.wanted_words else '_unknown_' for y in y_true]
        y_pred = [y if y in self.wanted_words else '_unknown_' for y in y_pred]
        wanted_words_confusion = confusion_matrix(y_true, y_pred)
        # classes = ["_silence_", "unknown", "yes", "on", "left", "go", "up", "off", "down", "right", "no", "stop"]
        # classes = ['_silence_', 'three', 'stop', 'five', 'seven', 'yes', 'bird', 'eight', 'dog', 'on', 'no', '\
        #            ' 'visual', 'tree', 'follow', 'one', 'sheila', 'happy', 'four', 'left', 'learn', '\
        #             ''go', 'zero', 'house', 'two', 'bed', 'up', 'off', 'six', 'marvin', 'down', '\
        #            ' 'forward', 'right', 'nine', 'cat', 'wow', 'backward']
        classes = ["_silence_", "Benjamin_Netanyau",  "Jens_Stoltenberg"  ,  "Julia_Gillard",  "Magaret_Tarcher"  ,
                   "Nelson_Mandela "]
        df_cm = pd.DataFrame(
            confusion, index=classes, columns=classes,
        )
        print(df_cm)
        wanted_accs = call_accuracies(wanted_words_confusion)
        acc_line = ('\n[%03d]: val_categorical_accuracy: %.2f, '
                    'val_mean_categorical_accuracy_wanted: %.2f') % (
                       epoch, acc, wanted_accs.mean())
        # with open('confusion_matrix.txt', 'a') as f:
        #   f.write('%s\n' % acc_line)
        #   f.write(confusion.to_dataframe().to_string())
        #
        # with open('wanted_confusion_matrix.txt', 'a') as f:
        #   f.write('%s\n' % acc_line)
        #   f.write(wanted_words_confusion.to_dataframe().to_string())

        logs['val_loss'] = val_loss
        logs['val_categorical_accuracy'] = acc
        logs['val_mean_categorical_accuracy_all'] = accs.mean()
        logs['val_mean_categorical_accuracy_wanted'] = wanted_accs.mean()
