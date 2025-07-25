import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import mplhep as hep

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from typing import List
from pathlib import Path


class Draw:
    def __init__(self,):
        hep.style.use("CMS")

    def plot_loss(
            self,
            name: str,
    ):
        log_file = f"models/{name}/{name}_training.log"
        df = pd.read_csv(log_file)

        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'], df['loss'], label='Training Loss', color='blue')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"models/{name}/{name}_loss.png")
        plt.clf()

    def plot_roc_curve(
            self,
            y_true,
            y_pred,
            name: str,
        ):
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, color='darkorange',
                    lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(f"models/{name}/{name}_roc.png")
            plt.clf()