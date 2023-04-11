import pickle as pkl
import numpy as np
from sklearn.utils import shuffle

def load_data(path):
    with open(path, 'rb') as pickle_file:
        all_data_prepared = pkl.load(pickle_file)
    print('data loaded: ', type(all_data_prepared))

    x_train = np.array(all_data_prepared['x_train'])
    x_train = np.expand_dims(x_train, axis=3)  # expand dimensions since grayscale
    y_train = np.array(all_data_prepared['y_train'])
    x_train, y_train = shuffle(x_train, y_train, random_state=0)  # shuffle before training
    x_test = np.array(all_data_prepared['x_test'])
    x_test = np.expand_dims(x_test, axis=3)  # expand dimensions since grayscale
    y_test = np.array(all_data_prepared['y_test'])
    x_val = np.array(all_data_prepared['x_val'])
    x_val = np.expand_dims(x_val, axis=3)  # expand dimensions since grayscale
    y_val = np.array(all_data_prepared['y_val'])

    return x_train, y_train, x_test, y_test, x_val, y_val
