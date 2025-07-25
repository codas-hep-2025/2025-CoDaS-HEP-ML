import tensorflow as tf
#tf.config.run_functions_eagerly(True)

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from keras_tuner.tuners import Hyperband

from pathlib import Path
from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, AveragePooling2D, Flatten, Input, Reshape, UpSampling2D, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping

from models import CNN, NN, CNN_NN, cut_pt, BDT, CNN_NN_opt
from drawing import Draw

# constants
epochs=10
loss="binary_crossentropy"

def main(args) -> None:

    test_features = np.load("test/features/preprocessed_cluster_features.npy")
    test_images = np.load("test/images/jet_images.npy")
    test_ids = np.load("test/ids/ids.npy")

    # mkdir for specified model
    if not os.path.isdir(f"models/{args.name}"):
        os.mkdir(f"models/{args.name}")

    model = load_model(f"models/{args.name}/{args.name}.keras")

    # Make submission
    if args.type == 'cnn': predictions = model.predict(test_images).flatten()
    elif args.type == 'nn': predictions = model.predict(test_features).flatten()
    elif args.type == 'cnn_nn': predictions = model.predict([test_images, test_features]).flatten()
    elif args.type == 'cut_pt': predictions = model.predict(test_features[:,[3]]).flatten()
    elif args.type == 'bdt': predictions = model.predict_proba(test_features)[:,1].flatten()
    elif args.type == 'cnn_nn_opt': predictions = model.predict([test_images, test_features]).flatten()
    solution = pd.DataFrame({'id': test_ids, 'label': predictions})
    solution.to_csv(f"models/{args.name}/solutions.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Kaggle Training""")
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="example",
        help="Name of model",
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="cnn_nn",
        help="Type of model; cnn, nn, cnn_nn, cut_pt, bdt, cnn_nn_opt",
    )
    parser.add_argument(
        "--rotate_reflect", "-rr",
        default=False,
        action="store_true",
        help="Include rotated/reflected data?",
    )
    parser.add_argument(
        "--optimize", "-o",
        default=False,
        action="store_true",
        help="Optimize with hyperband?",
    )
    main(parser.parse_args())
