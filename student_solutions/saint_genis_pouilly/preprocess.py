import numpy as np
import pandas as pd
import h5py

# for each set
for set in ["train", "val", "test"]:

    # for features first

    # load data
    features = pd.read_csv(f"{set}/features/cluster_features.csv")

    # we care about differences in eta/phi (mod 2pi), not values themselves
    features['diff_cluster_eta'] = features["max_cluster_eta"].to_numpy() - features["mean_cluster_eta"].to_numpy()
    features['diff_cluster_phi'] = (features["max_cluster_phi"].to_numpy() - features["mean_cluster_phi"].to_numpy()) % (2 * np.pi) - np.pi
    
    # get rid of noninformative features, add new features
    features = features[[
        "n_clusters",
        "mean_cluster_pt",
        "std_cluster_pt",
        "total_pt",
        "max_cluster_eta",
        "max_cluster_phi",
        "mean_cluster_eta",
        "mean_cluster_phi",
        "diff_cluster_eta",
        "diff_cluster_phi",
        "cluster_pt_ratio",
    ]]

    # save
    features = features.to_numpy()
    np.save(f"{set}/features/preprocessed_cluster_features.npy", features)

    # now for .h5 images

    # load data and convert to np
    images_h5 = h5py.File(f"{set}/images/jet_images.h5")
    images = images_h5["images"]
    images = np.array(images).reshape((-1, 30, 30, 1))

    # save
    np.save(f"{set}/images/jet_images.npy", images)
