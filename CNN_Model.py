import pickle as pkl
import numpy as np
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.callbacks import History
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

def dataloader(path):
    with open(path, 'rb') as pickle_file:
        all_data_prepared = pkl.load(pickle_file)
    print('data loaded: ', type(all_data_prepared))

    x_train = np.array(all_data_prepared['x_train'])
    y_train = np.array(all_data_prepared['y_train'])
    x_test = np.array(all_data_prepared['x_test'])
    y_test = np.array(all_data_prepared['y_test'])
    x_val = np.array(all_data_prepared['x_val'])
    y_val = np.array(all_data_prepared['y_val'])

    return x_train, y_train, x_test, y_test, x_val, y_val

def cnn_model():
    input_layer = Input(shape=(128, 128, 3))
    x = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same', activation='relu', input_shape=(128,128,3))(input_layer)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='same', activation='relu')(x)
    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    print(model.summary())

    return model

def train_model(x_train, y_train, model):

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    loss = BinaryCrossentropy()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    
    history = model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=2, shuffle=True, verbose=1)

    return history, model


def main():
    # load data
    print('loading data')
    x_train, y_train, x_test, y_test, x_val, y_val = dataloader(
        'C:/MyStuff/Kaggle_Practise/datasets/pneumonia_xray/all_data_prepared.pickle')

    # define model
    model = cnn_model()

    # start training
    print('start training')
    history, model = train_model(x_train, y_train, model)
    print('training finished')

    # print main KPIs
    print(history.history['acc'])
    print(history.history['val_acc'])
    print(history.history['loss'])
    print(history.history['val_loss'])

    # Plot history: Loss
    plt.plot(history.history['loss'], label='BinaryCrossentropy (train data)')
    plt.plot(history.history['val_loss'], label='BinaryCrossentropy (validation data)')
    plt.title('BinaryCrossentropy XRays Trainingsset')
    plt.ylabel('BinaryCrossentropy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

    # Plot history: ACC
    plt.plot(history.history['acc'], label='Accuracy (train data)')
    plt.plot(history.history['val_acc'], label='Accuracy (validation data)')
    plt.title('Accuracy XRays Trainingsset')
    plt.ylabel('Accuracy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()