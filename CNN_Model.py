import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input, BatchNormalization
from keras.callbacks import History, ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from tensorflow.math import confusion_matrix
from sklearn.utils import shuffle

def dataloader(path):
    with open(path, 'rb') as pickle_file:
        all_data_prepared = pkl.load(pickle_file)
    print('data loaded: ', type(all_data_prepared))

    x_train = np.array(all_data_prepared['x_train'])
    y_train = np.array(all_data_prepared['y_train'])
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test = np.array(all_data_prepared['x_test'])
    y_test = np.array(all_data_prepared['y_test'])
    x_val = np.array(all_data_prepared['x_val'])
    y_val = np.array(all_data_prepared['y_val'])

    return x_train, y_train, x_test, y_test, x_val, y_val

def cnn_model(shape):
    input_layer = Input(shape=shape)
    x = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same', activation='relu', input_shape=shape)(input_layer)
    x = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same', activation='relu', input_shape=shape)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.5)(x) 

    x = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.5)(x)

    x = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.5)(x)

    x = Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(1, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())

    return model

def train_model(x_train, y_train, model, epochs, batch_size, path):

    # define callbacks
    checkpoint = ModelCheckpoint(path, save_weights_only=True, monitor='val_acc', mode='max',
                 save_best_only=True)

    # define parameters
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    loss = BinaryCrossentropy()
    
    # initiate training
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    
    # start training
    history = model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2,
                        epochs=epochs, shuffle=False, verbose=1, callbacks=[checkpoint])

    return history, model

def eval_model(model, x_test, y_test, batch_size):

    # evaluate model on test set
    evaluation = model.evaluate(x_test, y_test, batch_size)
    print('test loss, test acc: ', evaluation)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print('Generate predictions for test set')
    y_pred = model.predict(x_test)

    # create and plot confusion matrx
    classes = ['0 = healthy', '1 = pneumenia']
    con_mat = confusion_matrix(labels=y_test, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat, index=classes, columns=classes)
    
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # load data
    print('loading data')
    x_train, y_train, x_test, y_test, x_val, y_val = dataloader(
        'C:/MyStuff/Kaggle_Practise/datasets/pneumonia_xray/all_data_prepared.pickle')

    # define model
    model = cnn_model(shape=np.shape(x_train[0]))

    # start training
    print('start training')
    history, model = train_model(x_train, y_train, model, epochs=20, batch_size=64,
                                 path='C:/MyStuff/Kaggle_Practise/models/pneumonia_xray/cnn_checkpoint.h5')
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

    # evaluate model
    model.load_weights('C:/MyStuff/Kaggle_Practise/models/pneumonia_xray/cnn_checkpoint.h5')
    eval_model(model, x_test, y_test, batch_size=64)

    # save model
    model.save('C:/MyStuff/Kaggle_Practise/models/pneumonia_xray/cnn_model.h5')


if __name__ == "__main__":
    main()