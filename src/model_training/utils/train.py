from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


def train_model(x_train, y_train, model, epochs, batch_size, path):
    # define callbacks
    checkpoint = ModelCheckpoint(
        path,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # define parameters
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    loss = BinaryCrossentropy()

    # initiate training
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # start training
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint]
    )

    return history, model
