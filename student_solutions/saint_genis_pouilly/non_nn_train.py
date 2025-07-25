import sys

from tensorflow import keras
import pandas as pd
import numpy as np

from drawing import Draw

def load_features(path, blacklist=None, means=None, stds=None):
    features = pd.read_csv(path)

    if blacklist is None:
        blacklist = features.columns[features.std() == 0]
        print("Blacklisted features:", blacklist)

    features_white = features.drop(blacklist, axis=1)

    if means is None:
        means = features_white.mean()
        stds = features_white.std()

    for col in features_white:
        features_white[col] = (features_white[col] - means[col]) / stds[col]

    return features_white, (blacklist, means, stds)

def expand_df(df, degree):
    powers = [df]

    for _ in range(degree-1):
        new_df = {}
        for comp_col in powers[-1]:
            for col in df:
                new_df[comp_col + "*" + col] = powers[-1][comp_col] * df[col]
        powers.append(pd.DataFrame(new_df))
    
    return pd.concat(powers, axis=1)

def train_perturbative(degree):
    train_features, training_meta = load_features("train/features/cluster_features.csv")
    val_features, _ = load_features("val/features/cluster_features.csv", *training_meta)
    
    train_labels = np.load("train/labels/labels.npy")
    val_labels = np.load("val/labels/labels.npy")

    assert len(train_labels) == len(train_features)
    assert len(train_features.keys()) == len(val_features.keys())
    assert len(val_labels) == len(val_features)

    train_features = expand_df(train_features, degree)
    val_features = expand_df(val_features, degree)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(len(train_features.keys()),)),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(train_features, train_labels, epochs=50, batch_size=8, validation_split=0.1)

    artist = Draw()
    val_pred = model.predict(val_features)
    artist.plot_roc_curve(val_labels, val_pred, "perturbative")

def main(argv):
    train_perturbative(3)
    

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))