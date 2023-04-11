import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow_addons.metrics import F1Score
# from tensorflow.keras.metrics import Accuracy, Precision, Recall


class MyHpyerModel(kt.HyperModel):
    def __init__(self, shape):
        self.shape = shape

    def build(self, hp):
        """
        Builds the model and sets the hyperparameter to tune

        Args:
            hp: Keras tuner object
            shape: tuple, shape of input layer

        Returns:
            model with hyperparameter to tune

        """
        # define set of hyperparameters
        hp_filters_layer_one = hp.Int(
            'units_layer_one',
            min_value=32,
            max_value=128,
            step=16
        )
        hp_filters_hidden_layers_batch_one = hp.Int(
            'units_hidden_layers_batch_one',
            min_value=16,
            max_value=64,
            step=16
        )
        hp_filters_hidden_layers_batch_two = hp.Int(
            'units_hidden_layers_batch_two',
            min_value=8,
            max_value=32,
            step=8
        )
        hp_kernel_size = hp.Choice(
            'kernel_size',
            values=[5, 3]
        )
        hp_stride = hp.Choice(
            'stride',
            values=[3, 2, 1]
        )
        hp_pool_method = hp.Choice(
            'pool_method',
            values=['avg', 'max', 'none']
        )
        hp_pool_size = hp.Choice(
            'pool_size',
            values=[3, 2]
        )
        hp_dropout_rate_conv = hp.Choice(
            'dropout_rate',
            values=[0.2, 0.0]
        )
        hp_dropout_rate_dense = hp.Choice(
            'dropout_rate',
            values=[0.5, 0.2]
        )
        # hp_regularization_rate = hp.Choice(
        #     'regularization_rate',
        #     values=[0.0001, 0.001, 0.01]
        # )
        hp_learning_rate = hp.Choice(
            'learning_rate',
            values=[0.0001, 0.001, 0.01, 0.1]
        )

        # define input and first layer
        input_layer = Input(shape=self.shape)
        x = Conv2D(
            filters=hp_filters_layer_one,
            kernel_size=hp_kernel_size,
            strides=(hp_stride, hp_stride),
            padding='same',
            activation='relu'
        )(input_layer)

        # add flexible number of layers
        for i in range(hp.Int('hidden_layers_batch_one', 1, 3)):
            x = Conv2D(
                filters=hp_filters_hidden_layers_batch_one,
                kernel_size=hp_kernel_size,
                strides=(hp_stride, hp_stride),
                padding='same',
                activation='relu'
            )(x)
            with hp.conditional_scope('pool_method', ['none']):
                if hp_pool_method == 'none':
                    pass
            with hp.conditional_scope('pool_method', ['avg']):
                if hp_pool_method == 'avg':
                    x = AveragePooling2D(pool_size=(hp_pool_size, hp_pool_size))(x)
            with hp.conditional_scope('pool_method', ['max']):
                if hp_pool_method == 'max':
                    x = MaxPool2D(pool_size=(hp_pool_size, hp_pool_size))(x)
            x = Dropout(rate=hp_dropout_rate_conv)(x)

        for i in range(hp.Int('hidden_layers_batch_two', 1, 3)):
            x = Conv2D(
                filters=hp_filters_hidden_layers_batch_two,
                kernel_size=hp_kernel_size,
                strides=(hp_stride, hp_stride),
                padding='same',
                activation='relu'
            )(x)
            x = Dropout(rate=hp_dropout_rate_conv)(x)

        # flatten
        x = Flatten()(x)

        # add two dense layers, one to shrink with additional params (weights), one as output layer
        x = Dense(units=4, activation='tanh')(x)
        x = Dropout(rate=hp_dropout_rate_dense)(x)
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
                F1Score(num_classes=1)
                # Accuracy(),
                # Precision(),
                # Recall()
            ]
        )

        print(model.summary())

        return model
