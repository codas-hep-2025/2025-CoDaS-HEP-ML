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

    # load data; 3520 training events, 502 validation events
    #train_labels = np.load("train/labels/labels.npy")
    #train_features = np.load("train/features/preprocessed_cluster_features.npy") # only 7 inputs
    #train_images = np.load("train/images/jet_images.npy") # 30 x 30 images
    #valid_labels = np.load("val/labels/labels.npy")
    #valid_features = np.load("val/features/preprocessed_cluster_features.npy")
    #valid_images = np.load("val/images/jet_images.npy")
    #valid_ids = np.load("val/ids/ids.npy")
    train_labels = np.load("train/labels/all.npy")
    train_features = np.load("train/features/all.npy") # only 7 inputs
    train_images = np.load("train/images/all.npy") # 30 x 30 images
    test_features = np.load("test/features/preprocessed_cluster_features.npy")
    test_images = np.load("test/images/jet_images.npy")
    test_ids = np.load("test/ids/ids.npy")

    # rotate/reflect image data. x60 increase in events
    if args.rotate_reflect:
        train_images, train_labels, train_features = rotate_reflect(train_images, train_labels, train_features)
        
    # mkdir for specified model
    if not os.path.isdir(f"models/{args.name}"):
        os.mkdir(f"models/{args.name}")

    # specify LR, callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    mc = ModelCheckpoint(f"models/{args.name}/{args.name}.keras", save_best_only=True)
    log = CSVLogger(f"models/{args.name}/{args.name}_training.log", append=True)
    callbacks = [mc, log, reduce_lr]

    # optimize
    if args.optimize:
        early_stop = EarlyStopping(monitor="val_loss", patience=5)
        if args.type == "cnn_nn":
            tuner = Hyperband(
                CNN_NN((30,30,1), (11,)).get_trial,
                objective="val_loss",
                max_epochs=epochs,
                factor=3,
                executions_per_trial=3,
                directory=f"{args.name}",
                project_name=f"{args.name}_tuning"
            )
        elif args.type == "cnn":
            tuner = Hyperband(
                CNN_NN((30,30,1), (11,)).get_trial,
                objective="val_loss",
                max_epochs=epochs,
                factor=3,
                executions_per_trial=3,
                directory=f"{args.name}",
                project_name=f"{args.name}_tuning"
            )
        tuner.search(
            [train_images, train_features], train_labels,
            validation_data=([valid_images, valid_features], valid_labels),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop]
        )
        tuner.search_space_summary(extended=False)

        # Get the best hyperparameters from the tuner
        best_hp = tuner.get_best_hyperparameters(1)[0]
        print("Best Hyperparameters:")
        for k, v in best_hp.values.items():
            print(f"  {k}: {v}")

        # Rebuild the model using best hyperparameters
        if args.type == "cnn_nn":
            model, batch_size = CNN_NN((30,30,1), (11,)).get_trial(best_hp)
        elif args.type == "cnn":
            model, batch_size = CNN((30,30,1), (11,)).get_trial(best_hp)

        # Train the model
        history = model.fit(
            [train_images, train_features], train_labels,
            validation_data=([valid_images, valid_features], valid_labels),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
        )

    else: # compile and train specified model
        if args.type == 'cnn':
            model = CNN((30, 30, 1)).get_model()
            model.fit(
                train_images, train_labels,
                epochs=epochs,
                validation_data=(valid_images, valid_labels),
                callbacks=callbacks,
                )
        elif args.type == 'nn':
            model = NN((11,)).get_model()
            model.fit(
                train_features, train_labels,
                epochs=epochs,
                validation_data = (valid_features, valid_labels),
                callbacks=callbacks,
            )
        elif args.type == 'cnn_nn':
            model = CNN_NN((30,30,1), (11,)).get_model()
            model.fit(
                [train_images, train_features], train_labels,
                epochs=epochs,
                #validation_data = ([valid_images, valid_features], valid_labels),
                callbacks=callbacks,
            )
        elif args.type == 'cut_pt':
            model = cut_pt((1,), optimizer=Adam(learning_rate=0.001), loss='mse').get_model()
            model.fit(
                train_features[:,[3]], train_labels,
                epochs=epochs,
                validation_data = (valid_features[:,[3]], valid_labels),
                callbacks=callbacks,
            )
        elif args.type == 'bdt':
            model = BDT(n_estimators=100, max_depth=5, learning_rate=0.1).get_model()
            model.fit(
                train_features, train_labels,
                eval_set = [(valid_features, valid_labels)],
                verbose=True,
            )
            model.get_booster().save_model(f"models/{args.name}/{args.name}.json")
        elif args.type == 'cnn_nn_opt':
            model = CNN_NN_opt((30,30,1), (11,)).get_model()
            model.fit(
                [train_images, train_features], train_labels,
                epochs=epochs,
                validation_data = ([valid_images, valid_features], valid_labels),
                callbacks=callbacks,
            )
    
    model = load_model(f"models/{args.name}/{args.name}.keras")

    # Write summary
    with open(f"models/{args.name}/{args.name}_summary.txt", "w", encoding="utf-8") as f:
        if args.type != 'bdt':
            model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Plot losses
    #draw = Draw()
    #if args.type != 'bdt':
    #    draw.plot_loss(args.name)
    
    # Plot roc curve, make submission on validation set
    #if args.type == 'cnn': predictions = model.predict(valid_images).flatten()
    #elif args.type == 'nn': predictions = model.predict(valid_features).flatten()
    #elif args.type == 'cnn_nn': predictions = model.predict([valid_images, valid_features]).flatten()
    #elif args.type == 'cut_pt': predictions = model.predict(valid_features[:,[3]]).flatten()
    #elif args.type == 'bdt': predictions = model.predict_proba(valid_features)[:,1].flatten()
    #elif args.type == 'cnn_nn_opt': predictions = model.predict([valid_images, valid_features]).flatten()
    #draw.plot_roc_curve(
    #    y_true=valid_labels,
    #    y_pred=predictions,
    #    name=args.name,
    #    )
    #solution = pd.DataFrame({'id': valid_ids, 'label': predictions})
    #solution.to_csv(f"models/{args.name}/valid_solutions.csv", index=False)

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
