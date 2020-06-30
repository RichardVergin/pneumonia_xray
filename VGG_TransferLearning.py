import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from keras.applications import vgg19
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
    x_train, y_train = shuffle(x_train, y_train, random_state=0) # shuffle before training
    x_test = np.array(all_data_prepared['x_test'])
    y_test = np.array(all_data_prepared['y_test'])
    x_val = np.array(all_data_prepared['x_val'])
    y_val = np.array(all_data_prepared['y_val'])

    return x_train, y_train, x_test, y_test, x_val, y_val

def transfer_model(shape):
    input_tensor = Input(shape=shape)

    base_model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)

    for layer in base_model.layers:
        layer.trainable = False

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block4_pool'].output
    
    # Build similar architecture from scratch model 
    x = Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(input=base_model.input, output=output_layer)
    print(model.summary())

    return model

def train_model(x_train, y_train, model, epochs, batch_size, path):

    # define callbacks
    checkpoint = ModelCheckpoint(path, save_weights_only=True, monitor='val_acc', mode='max',
                 save_best_only=True, verbose=1)

    # define parameters
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    loss = BinaryCrossentropy()
    
    # initiate training
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    
    # start training
    history = model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2,
                        epochs=epochs, shuffle=True, verbose=1, callbacks=[checkpoint])

    return history, model

def eval_model(model, x_test, y_test, batch_size):

    # evaluate model on test set
    evaluation = model.evaluate(x_test, y_test, batch_size)
    print('test loss, test acc: ', evaluation)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print('Generate predictions for test set')
    y_pred = model.predict(x_test)
    print('predictions on test set: {}'.format(y_pred))

    # round probabilities to 0 and 1
    y_rounded = y_pred
    for prediction in range(len(y_rounded)):
        if y_rounded[prediction] >= 0.5:
            y_rounded[prediction] = 1
        else:
            y_rounded[prediction] = 0

    return y_pred, y_rounded

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')

    print(cm)

    tresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment='center',
        color='white' if cm[i, j] > tresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # load data
    print('loading data')
    x_train, y_train, x_test, y_test, x_val, y_val = dataloader(
        'C:/MyStuff/Kaggle_Practise/datasets/pneumonia_xray/all_data_prepared_rgb.pickle')

    # define model
    print('our shape = {}'.format(np.shape(x_train[0])))
    model = transfer_model(shape=np.shape(x_train[0]))

    # start training
    print('start training')
    history, model = train_model(x_train, y_train, model, epochs=50, batch_size=32,
                                 path='C:/MyStuff/Kaggle_Practise/models/pneumonia_xray/transfer_checkpoint_rgb.h5')
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
    model.load_weights('C:/MyStuff/Kaggle_Practise/models/pneumonia_xray/transfer_checkpoint_rgb.h5')
    y_pred, y_rounded = eval_model(model, x_test, y_test, batch_size=32)
    cm =  confusion_matrix(y_test, y_rounded)
    cm_plot_labels = ['No Pneumonia', 'Pneumonia']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix prediction test set')


    # save model
    model.save('C:/MyStuff/Kaggle_Practise/models/pneumonia_xray/transfermodel_rgb.h5')


if __name__ == "__main__":
    main()
