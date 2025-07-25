from sklearn.model_selection import KFold
import numpy as np
import os

train_features = np.load("train/features/preprocessed_cluster_features.npy")
train_images = np.load("train/images/jet_images.npy")
valid_features = np.load("val/features/preprocessed_cluster_features.npy")
valid_images = np.load("val/images/jet_images.npy")
train_labels = np.load("train/labels/labels.npy")
valid_labels = np.load("val/labels/labels.npy")
train_ids = np.load("train/ids/ids.npy")
valid_ids = np.load("val/ids/ids.npy")

features = np.concatenate([train_features, valid_features], axis=0)
images = np.concatenate([train_images, valid_images], axis=0)
labels = np.concatenate([train_labels, valid_labels], axis=0)
ids = np.concatenate([train_ids, valid_ids], axis=0)

np.save(f"train/features/all.npy", features)
np.save(f"train/images/all.npy", images)
np.save(f"train/labels/all.npy", labels)
np.save(f"train/ids/all.npy", ids)

kf = KFold(n_splits=8, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
    np.save(f"train/features/cross_preprocessed_cluster_features_{fold}.npy", features[train_idx])
    np.save(f"train/images/cross_jet_images_{fold}.npy", images[train_idx])
    np.save(f"train/labels/cross_labels_{fold}.npy", labels[train_idx])
    np.save(f"train/ids/cross_ids_{fold}.npy", ids[train_idx])

    np.save(f"val/features/cross_preprocessed_cluster_features_{fold}.npy", features[val_idx])
    np.save(f"val/images/cross_jet_images_{fold}.npy", images[val_idx])
    np.save(f"val/labels/cross_labels_{fold}.npy", labels[val_idx])
    np.save(f"val/ids/cross_ids_{fold}.npy", ids[val_idx])