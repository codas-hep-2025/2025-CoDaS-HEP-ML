import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def main(args) -> None:
    
    # load predictions, labels, and images
    predictions = pd.read_csv(f"models/{args.name}/valid_solutions.csv")
    labels = np.load("val/labels/labels.npy")
    images = np.load("val/images/jet_images.npy")

    
    # Combine into single DataFrame
    df = pd.DataFrame({
        'pred': predictions['label'].values,
        'true': labels
    })
    df['correct'] = (df['pred'].round() == df['true'])
    df['confidence'] = np.abs(df['pred'] - 0.5)
    df['idx'] = df.index  # to track back to image

    # Define four categories
    correct_1 = df[(df['correct']) & (df['true'] == 1)].sort_values(by='pred', ascending=False).head(6)
    correct_0 = df[(df['correct']) & (df['true'] == 0)].sort_values(by='pred', ascending=True).head(6)
    incorrect_1 = df[(~df['correct']) & (df['true'] == 0)].sort_values(by='pred', ascending=False).head(6)
    incorrect_0 = df[(~df['correct']) & (df['true'] == 1)].sort_values(by='pred', ascending=True).head(6)
    uncertain_1 = df[(df["true"] == 1) & (df["pred"].between(0.45, 0.55))].sort_values("pred").head(6)
    uncertain_0 = df[(df["true"] == 0) & (df["pred"].between(0.45, 0.55))].sort_values("pred").head(6)

    # Plotting helper
    def plot_examples(subset_df, images, filename):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # Widen canvas
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        im = None
        for j, ax in enumerate(axes.flatten()):
            if j >= len(subset_df):
                ax.axis('off')
                continue
            row = subset_df.iloc[j]
            img = images[row['idx'], :, :, 0]
            img = np.sqrt(np.sqrt(img))
            im = ax.imshow(img, cmap='viridis', origin='lower', vmin=0, vmax=math.sqrt(2))
            ax.set_title(f"Pred: {row['pred']:.2f}, True: {int(row['true'])}")
            ax.axis('off')
            ax.set_xlabel("eta")
            ax.set_ylabel("phi")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on the right for colorbar
        plt.savefig(filename)
        plt.close()

    # Save all four categories
    plot_examples(correct_1, images, f"models/{args.name}/correct_1.png")
    plot_examples(correct_0, images, f"models/{args.name}/correct_0.png")
    plot_examples(incorrect_1, images, f"models/{args.name}/incorrect_0.png")
    plot_examples(incorrect_0, images, f"models/{args.name}/incorrect_1.png")
    plot_examples(uncertain_1, images, f"models/{args.name}/uncertain_1.png")
    plot_examples(uncertain_0, images, f"models/{args.name}/uncertain_0.png")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Kaggle Inspect Events""")
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="example",
        help="Name of model",
    )
    main(parser.parse_args())
