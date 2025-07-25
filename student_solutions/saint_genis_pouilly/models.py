import tensorflow as tf
import numpy as np
import xgboost as xgb

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda, 
    Reshape,
    UpSampling2D,
    Conv2DTranspose,
    ZeroPadding1D,
    Cropping2D, 
    MaxPooling2D,
)
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperParameters

class Autoencoder:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(30, (3, 3), strides=1, padding="same")(inputs)
        x = Activation("relu", name="relu_1")(x)
        x = AveragePooling2D((2, 2), name="pool_1")(x)
        x = Conv2D(40, (3, 3), strides=1, padding="same")(x)
        x = Activation("relu", name="relu_1")(x)
        x = Flatten(name="flatten")(x)
        x = Dense(100, activation="relu")(x)
        x = Dense(15*15*30)(x)
        x = Reshape((15, 15, 30))(x)
        x = Activation("relu", name="relu_-1")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = Conv2DTranspose(30, (3, 3), strides=2, padding="same")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        outputs = Conv2D(1, (3, 3), activation="relu", strides=1, padding="same")(x)
        return Model(inputs, outputs, name="Autoencoder")
    

class MixedPadConv2D(tf.keras.layers.Layer):
    """
    Circular on the 1st axis, zeros on the 0th axis, assuming batched input
    """
    def __init__(self, filters, kernelSize, **kwargs):
        super().__init__()
        self.k = kernelSize
        self.conv = tf.keras.layers.Conv2D(filters, kernelSize, padding='valid', **kwargs)
    
    def call(self, x):
        ph = self.k[0] // 2
        pw = self.k[1] // 2
        x = tf.pad(x, [[0,0],[ph,ph],[0,0],[0,0]], mode='CONSTANT') # zero pad height
        x = tf.concat([x[:,:,-pw:,:], x, x[:,:,:pw,:]], axis=2)     # manual circular pad width
        return self.conv(x)
    
class CNN:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.optimizer = Adam(learning_rate=0.001)
        self.loss = "binary_crossentropy"

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        x = MixedPadConv2D(16, (3, 3), strides=1)(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
        x = MixedPadConv2D(16, (3, 3), strides=1)(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
        x = Flatten(name="flatten")(x)
        x = Dense(16, activation="relu")(x)
        x = Dense(16, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs, name="CNN")
        model.compile(optimizer=Adam(), loss=self.loss)
        return model

class NN:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.optimizer = Adam(learning_rate=0.001)
        self.loss = "binary_crossentropy"

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        x = Dense(11, activation="relu")(inputs)
        x = Dense(11, activation="relu")(x)
        x = Dense(11, activation="relu")(x)
        x = Dense(11, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs, name="NN")
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

class cut_pt:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.optimizer = Adam(learning_rate=0.001)
        self.loss = "binary_crossentropy"

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        outputs = Dense(1, activation="sigmoid")(inputs)
        model = Model(inputs, outputs, name="cut_pt")
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model
    
class CNN_NN_opt:
    def __init__(self, input_shape_cnn: tuple, input_shape_nn: tuple):
        self.input_shape_cnn = input_shape_cnn
        self.input_shape_nn = input_shape_nn
        self.optimizer = Adam(learning_rate=0.003)
        self.loss = "binary_crossentropy"

    def get_model(self):
        # CNN input
        inputs_cnn = Input(shape=self.input_shape_cnn, name="cnn_input")
        x = inputs_cnn

        # Convolutional layers
        x = Conv2D(64, (5, 5), padding="same")(x)
        x = Activation("relu")(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        x = Flatten(name="flatten")(x)

        # Dropout after flatten
        x = Dropout(0.3)(x)

        # NN input
        inputs_nn = Input(shape=self.input_shape_nn, name="nn_input")

        # Combine CNN and NN branches
        x = Concatenate(axis=-1)([x, inputs_nn])

        # Dense layers after concatenation
        x = Dense(128, activation="relu")(x)

        x = Dropout(0.3)(x)

        # Output layer
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=[inputs_cnn, inputs_nn], outputs=outputs, name="CNN_NN")
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

class CNN_NN:
    def __init__(self, input_shape_cnn: tuple, input_shape_nn: tuple):
        self.input_shape_cnn = input_shape_cnn
        self.input_shape_nn = input_shape_nn
        self.optimizer = Adam(learning_rate=0.001)
        self.loss = "binary_crossentropy"

    def get_model(self):

        # CNN branch
        inputs_cnn = Input(shape=self.input_shape_cnn)
        x = Conv2D(8, (5, 5), padding="same")(inputs_cnn)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2))(x)

        x = Conv2D(8, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dropout(0.25)(x)

        # Dense feature branch
        inputs_nn = Input(shape=self.input_shape_nn)

        # Merge
        x = Concatenate()([x, inputs_nn])
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(8, activation="relu")(x)
        x = Dropout(0.1)(x)

        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=[inputs_cnn, inputs_nn], outputs=outputs, name="CNN_NN")
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

    def get_trial(self, hp: HyperParameters):
        # Set LR
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        self.optimizer = Adam(learning_rate=learning_rate)

        # CNN input
        inputs_cnn = Input(shape=self.input_shape_cnn, name="cnn_input")
        x = inputs_cnn

        # Convolutional layers
        for i in range(hp.Int("num_conv_layers", 1, 3)):
            filters = hp.Int(f"filters_{i}", min_value=8, max_value=64, step=8)
            kernel_size = hp.Choice(f"kernel_size_{i}", values=[3, 5])
            x = Conv2D(filters, kernel_size, padding="same")(x)
            x = Activation("relu")(x)
            x = AveragePooling2D(pool_size=(2, 2))(x)

        x = Flatten(name="flatten")(x)

        # Dropout after flatten
        dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
        if dropout_rate > 0.0:
            x = Dropout(dropout_rate)(x)

        # NN input
        inputs_nn = Input(shape=self.input_shape_nn, name="nn_input")

        # Combine CNN and NN branches
        x = Concatenate(axis=-1)([x, inputs_nn])

        # Dense layers after concatenation
        for i in range(hp.Int("num_dense_layers", 1, 3)):
            units = hp.Int(f"dense_units_{i}", min_value=8, max_value=128, step=8)
            x = Dense(units, activation="relu")(x)

        # Optional dropout after final dense
        final_dropout = hp.Float("final_dropout", 0.0, 0.5, step=0.1)
        if final_dropout > 0.0:
            x = Dropout(final_dropout)(x)

        # Output layer
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=[inputs_cnn, inputs_nn], outputs=outputs, name="CNN_NN_Tuned")
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

class BDT:
    def __init__(self, n_estimators: int, max_depth: int, learning_rate: float):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def get_model(self):
        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,  # Number of boosting rounds
            max_depth=self.max_depth,      # Maximum tree depth
            learning_rate=self.learning_rate, # Step size shrinkage
            objective='binary:logistic',  # Binary classification
            random_state=42,
        )
        return model
