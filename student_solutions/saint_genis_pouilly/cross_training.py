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
cross=8
epochs=50
loss="binary_crossentropy"

def rotate_reflect(images, labels, features): # rotate and reflect image-like data. Does not affect feature-like data.
    images = list(images)
    labels = list(labels)
    features = list(features)
    images2 = [i for i in images]
    for i, val in enumerate(images2):
        for j in range(1,30):
            rot = np.concat([val[:,j:], val[:,:j]], axis=1)
            images.append(rot)
            labels.append(labels[i])
            features.append(features[i])
            images.append(rot[::-1]) #reverse
            labels.append(labels[i])
            features.append(features[i])
    return np.array(images), np.array(labels), np.array(features)

def main(args) -> None:

    draw = Draw()

    # load data; 3520 training events, 502 validation events
    train_labels, train_features, train_images, train_ids, valid_labels, valid_features, valid_images, valid_ids, test_features, test_images, test_ids = [], [], [], [], [], [], [], [], [], [], []
    for i in range(cross):
        train_labels.append(np.load(f"train/labels/cross_labels_{i}.npy"))
        train_features.append(np.load(f"train/features/cross_preprocessed_cluster_features_{i}.npy"))
        train_images.append(np.load(f"train/images/cross_jet_images_{i}.npy"))
        train_ids.append(np.load(f"train/ids/cross_ids_{i}.npy"))
        valid_labels.append(np.load(f"val/labels/cross_labels_{i}.npy"))
        valid_features.append(np.load(f"val/features/cross_preprocessed_cluster_features_{i}.npy"))
        valid_images.append(np.load(f"val/images/cross_jet_images_{i}.npy"))
        valid_ids.append(np.load(f"val/ids/cross_ids_{i}.npy"))
    test_features = np.load("test/features/preprocessed_cluster_features.npy")
    test_images = np.load("test/images/jet_images.npy")
    test_ids = np.load("test/ids/ids.npy")

    # rotate/reflect image data. x60 increase in events
    if args.rotate_reflect:
        for i in range(cross):
            train_images[i], train_labels[i], train_features[i] = rotate_reflect(train_images[i], train_labels[i], train_features[i])
        
    # mkdir for specified model
    if not os.path.isdir(f"models/{args.name}"):
        os.mkdir(f"models/{args.name}")

    predictions, ids, labels, test_predictions = [], [], [], []
    # compile and train specified model
    for i in range(cross):

        ###
        
        model = load_model(f"models/{args.name}/{args.name}_{i}.keras")
        
        predictions.append(model.predict([valid_images[i], valid_features[i]]).flatten())
        test_predictions.append(model.predict([test_images, test_features]).flatten())
        ids.append(valid_ids[i])
        labels.append(valid_labels[i])

        #draw.plot_loss(args.name)
    
    predictions = np.concatenate(predictions)
    ids = np.concatenate(ids)
    labels = np.concatenate(labels)
    test_predictions = np.array(test_predictions)
    test_predictions = np.mean(test_predictions, axis=0).flatten()

    # Write summary
    with open(f"models/{args.name}/{args.name}_summary.txt", "w", encoding="utf-8") as f:
        if args.type != 'bdt':
            model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Plot roc curve, make submission on validation set
    draw.plot_roc_curve(
        y_true=labels,
        y_pred=predictions,
        name=args.name,
        )

    # Make submission
    solution = pd.DataFrame({'id': test_ids, 'label': test_predictions})
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
