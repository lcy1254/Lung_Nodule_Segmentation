'Plot in real-time'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

import time, random
import math
import os.path
import itertools
import csv

import numpy as np

from tensorflow.keras.callbacks import Callback

class MetricsHistory(Callback):
    """
    Tracks accuracy and loss in real-time, and plots it.
    Parameters
    ----------
    saving_path : string (optional, default is 'test.csv')
        Full path to output csv file.
    """
    def __init__(self, saving_path='test.csv'):
        # Initialization
        self.saving_path = saving_path
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs={}):
        # Store
        self.epochs += [epoch]
        self.losses += [logs.get('loss')]
        self.val_losses += [logs.get('val_loss')]
        self.accs += [logs.get('accuracy')]
        self.val_accs += [logs.get('val_accuracy')]

        # Save to file
        dictionary = {'epochs': self.epochs,
                      'losses': self.losses,
                      'val_losses': self.val_losses,
                      'accs': self.accs,
                      'val_accs': self.val_accs}
        dict_to_csv(dictionary, self.saving_path)

def dict_to_csv(dictionary, path):
    'Saves Python dictionary to a csv file'
    w = csv.writer(open(path, 'w+'))
    for key, val in dictionary.items():
        w.writerow([key, val])
