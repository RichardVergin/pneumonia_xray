import pickle as pkl
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input, BatchNormalization


def dataloader(path):
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


def build_model(shape):
    input_layer = Input(shape=shape)
    x = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same', activation='relu', input_shape=shape)(input_layer)
    x = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same', activation='relu', input_shape=shape)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.5)(x) 

    x = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.5)(x)

    # x = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool2D(pool_size=(2,2))(x)
    # x = Dropout(rate=0.5)(x)

    x = Conv2D(filters=1, kernel_size=1, strides=   (1, 1), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='sigmoid')(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())

    return model
