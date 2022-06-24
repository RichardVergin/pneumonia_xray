import keras_tuner as kt


def model_builder(hp):
    """
    Builds the model and sets the hyperparameter to tune

    Args:
        hp: Keras tuner object
        shape: tuple, shape of input layer

    Returns:
        model with hyperparameter to tune

    """
    # define set of hyperparameters
    hp_units_layer_one = hp.Int(
        'units_layer_one',
        min_value=16,
        max_value=64,
        step=16
    )
    hp_units_hidden_layers_batch_one = hp.Int(
        'units_hidden_layers_batch_one',
        min_value=16,
        max_value=64,
        step=16
    )
    hp_units_hidden_layers_batch_two = hp.Int(
        'units_hidden_layers_batch_two',
        min_value=8,
        max_value=32,
        step=8
    )
    hp_dropout_rate = hp.Choice(
        'dropout_rate',
        values=[0.5, 0.2]
    )
    hp_regularization_rate = hp.Choice(
        'regularization_rate',
        values=[0.0001, 0.001, 0.01]
    )
    hp_learning_rate = hp.Choice(
        'learning_rate',
        values=[0.0001, 0.001, 0.01, 0.1]
    )

    # define input and first layer
    input_layer = Input(shape=input_shape)
    x = Dense(units=hp_units_layer_one, activation='tanh')(input_layer)

    # add flexible number of layers
    for i in range(hp.Int('hidden_layers_batch_one', 1, 3)):
        x = Dense(
            units=hp_units_hidden_layers_batch_one,
            kernel_regularizer=regularizers.l1_l2(l1=hp_regularization_rate, l2=hp_regularization_rate),
            bias_regularizer=regularizers.l2(hp_regularization_rate),
            activity_regularizer=regularizers.l2(hp_regularization_rate),
            activation='tanh'
        )(x)
        # x = Dropout(rate=hp_dropout_rate)(x)

    for i in range(hp.Int('hidden_layers_batch_two', 1, 3)):
        x = Dense(
            units=hp_units_hidden_layers_batch_two,
            kernel_regularizer=regularizers.l1_l2(l1=hp_regularization_rate, l2=hp_regularization_rate),
            bias_regularizer=regularizers.l2(hp_regularization_rate),
            activity_regularizer=regularizers.l2(hp_regularization_rate),
            activation='tanh'
        )(x)
        x = Dropout(rate=hp_dropout_rate)(x)

    # define output layer
    output_layer = Dense(units=1, activation='sigmoid')(x)

    # define model
    model = Model(inputs=input_layer, outputs=output_layer)

    # compile model
    optimizer = Adam(
        learning_rate=hp_learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=[
            AUC(),
            Precision(),
            Recall()
        ]
    )

    print(model.summary())

    return model
