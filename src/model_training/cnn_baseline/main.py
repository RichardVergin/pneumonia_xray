from src.config import DATA_DIR, MODEL_DIR, PLOT_DIR
from src.model_training.utils.load_data import load_data
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.model_training.cnn_baseline.model import build_model
from src.model_training.utils.train import train_model
from src.model_training.utils.eval import keras_model_evaluation, plot_confusion_matrix


def main():
    path_prepared_data = DATA_DIR / 'all_data_prepared.pickle'
    path_checkpoint = MODEL_DIR / 'cnn_baseline' / 'cnn_checkpoint.h5'
    path_model = MODEL_DIR / 'cnn_baseline' / 'cnn_model.h5'
    folder_plots = PLOT_DIR / 'cnn_baseline'
    if not os.path.exists(str(folder_plots)):
        os.makedirs(str(folder_plots))
    else:
        pass

    # load data
    print('loading data')
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(str(path_prepared_data))

    # define model
    model = build_model(shape=np.shape(x_train[0]))

    # check if path for checkpoints is available
    if not os.path.exists(
            os.path.dirname(str(path_checkpoint))
    ):
        os.makedirs(
            os.path.dirname(str(path_checkpoint))
        )
    else:
        pass

    # start training
    print('start training')
    history, model = train_model(x_train, y_train, model, epochs=25, batch_size=32, path=str(path_checkpoint))
    print('training finished')

    # print main KPIs
    print(history.history['acc'])
    print(history.history['val_acc'])
    print(history.history['loss'])
    print(history.history['val_loss'])

    # Plot and save history: Loss
    plt.plot(history.history['loss'], label='BinaryCrossentropy (train data)')
    plt.plot(history.history['val_loss'], label='BinaryCrossentropy (validation data)')
    plt.title('BinaryCrossentropy XRays Trainingsset')
    plt.ylabel('BinaryCrossentropy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig(folder_plots / 'development_of_loss')
    plt.show()

    # Plot history: ACC
    plt.plot(history.history['acc'], label='Accuracy (train data)')
    plt.plot(history.history['val_acc'], label='Accuracy (validation data)')
    plt.title('Accuracy XRays Trainingsset')
    plt.ylabel('Accuracy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig(folder_plots / 'development_of_acc')
    plt.show()

    # evaluate model
    model.load_weights(str(path_checkpoint))
    y_pred, y_rounded = keras_model_evaluation(model, x_test, y_test, batch_size=32)
    cm = confusion_matrix(y_test, y_rounded)
    cm_plot_labels = ['No Pneumonia', 'Pneumonia']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix prediction test set')
    plt.savefig(folder_plots / 'confusion_matrix')

    # save model
    if not os.path.exists(
            os.path.dirname(str(path_model))
    ):
        os.makedirs(
            os.path.dirname(str(path_model))
        )
    else:
        pass
    model.save(str(path_model))


if __name__ == "__main__":
    main()
