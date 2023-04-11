from src.config import DATA_DIR, MODEL_DIR, PLOT_DIR
from src.model_training.utils.load_data import load_data
import numpy as np
import os
import keras_tuner as kt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.model_training.cnn_tuning.model import MyHpyerModel
from src.model_training.utils.eval import keras_model_evaluation, plot_confusion_matrix


def main():
    path_prepared_data = DATA_DIR / 'all_data_prepared.pickle'
    path_checkpoint = MODEL_DIR / 'cnn_tuning' / 'cnn_checkpoint.h5'
    path_model = MODEL_DIR / 'cnn_tuning' / 'cnn_model.h5'
    folder_plots = PLOT_DIR / 'cnn_tuning'
    if not os.path.exists(str(folder_plots)):
        os.makedirs(str(folder_plots))
    else:
        pass

    # load data
    print('loading data')
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(str(path_prepared_data))

    # initialize model and random search
    tuner = kt.RandomSearch(
        hypermodel=MyHpyerModel(
            shape=np.shape(x_train[0])
        ),
        objective='val_loss',
        max_trials=10,
        seed=42
        # overwrite=True
    )

    # check if path for checkpoints is available
    if not os.path.exists(
            os.path.dirname(str(path_checkpoint))
    ):
        os.makedirs(
            os.path.dirname(str(path_checkpoint))
        )
    else:
        pass

    # define callback
    checkpoint = ModelCheckpoint(
        filepath=str(path_checkpoint),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    earlystop = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=4,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    )

    # start training
    print('start hyperparameter tuning')
    history = tuner.search(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=25,
        verbose=1,
        validation_split=0.2,
        shuffle=True,
        callbacks=[checkpoint, earlystop]
    )
    print('hyperparameter tuning finished')

    # get best hyperparameters and refit model
    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=3,
        verbose=1,
        validation_split=0.2,
        shuffle=True,
        callbacks=[checkpoint]
    )

    # print main KPIs
    print((history.history['f1_score']))
    print(history.history['val_f1_score'])
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
    plt.plot(history.history['f1_score'], label='F1 Score (train data)')
    plt.plot(history.history['val_f1_score'], label='F1 Score (validation data)')
    plt.title('F1 Score XRays Trainingsset')
    plt.ylabel('F1 Score value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig(folder_plots / 'development_of_f1_score')
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
