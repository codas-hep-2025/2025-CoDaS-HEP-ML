# spconv_ss3l_2d_model.py
# SpConv Single Stream 3-Layer CNN using 2D convolutions with layers as channels

import spconv.pytorch as spconv
from torch import nn
from spconv.pytorch import SparseMaxPool2d


class CalorimeterCNN(nn.Module):
    """
    SpConv version of Single Stream 3-Layer CNN architecture

    Uses sparse 2D convolutions on calorimeter data (64×64 grid)
    with 3 layers treated as input channels

    Args:
        input_shape: (layers, height, width) - default (3, 64, 64)
    """

    def __init__(self, input_shape=(3, 64, 64)):
        super().__init__()

        self.n_layers = input_shape[0]  
        self.spatial_shape = input_shape[1:]  # (64, 64)

        # Sparse Conv Block 1: Process 3 input channels
        self.sparse_conv1 = spconv.SparseSequential(
            spconv.SparseConv2d(self.n_layers, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SparseMaxPool2d(kernel_size=2, stride=2),  # 32->16
        )

        # Sparse Conv Block 2
        self.sparse_conv2 = spconv.SparseSequential(
            spconv.SparseConv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SparseMaxPool2d(kernel_size=2, stride=2),  # 16->8
        )

        # Sparse Conv Block 3: Convert to dense after this
        self.sparse_conv3 = spconv.SparseSequential(
            spconv.SparseConv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SparseMaxPool2d(kernel_size=2, stride=2),  
            spconv.ToDense(),  
        )

        # Calculate flattened size after convolutions
        self.flat_features = 128 * 8 * 8

        # Dropout layers (applied after sparse→dense conversion)
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.2)

        self.shared_fc = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # ALP vs photon head

        self.classification_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        # mass is a distribution so likely not bounded
        self.mass_regression_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        print("SpConv SS3L 2D model initialized")
        print(f"  Input channels (layers): {self.n_layers}")
        print(f"  Spatial shape: {self.spatial_shape}")
        print(f"  Flattened features: {self.flat_features}")

    def forward(self, sparse_tensor):
        """
        Forward pass through SpConv 2D layers

        Args:
            sparse_tensor: spconv.SparseConvTensor with multi-channel features

        Returns:
            torch.Tensor: [batch_size, 1] prediction scores
        """
        # Process through sparse convolutions
        x = self.sparse_conv1(sparse_tensor)

        # Apply dropout to intermediate dense representation if needed

        x = self.sparse_conv2(x)
        x = self.sparse_conv3(x)
        # Apply dropout to dense output
        x = self.dropout3(x)

        # Flatten: [batch, 128, 8, 8] → [batch, 8192]
        x = x.view(x.size(0), -1)

        shared_features = self.shared_fc(x)
        classification_output = self.classification_head(shared_features)
        mass_output = self.mass_regression_head(shared_features)
        return classification_output, mass_output

    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        shared_params = sum(p.numel() for p in self.shared_fc.parameters())
        class_params = sum(p.numel() for p in self.classification_head.parameters())
        mass_params = sum(p.numel() for p in self.mass_regression_head.parameters())
        conv_params = total_params - shared_params - class_params - mass_params
        return {
            "model_name": "SpConv_SS3L_2D",
            "input_channels": self.n_layers,
            "spatial_shape": self.spatial_shape,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "conv_parameters": conv_params,
            "shared_fc_parameters": shared_params,
            "classification_head_parameters": class_params,
            "mass_regression_head_parameters": mass_params,
            "architecture": "Multi-Task 2D Sparse CNN with shared backbone",
            "tasks": ["binary_classification", "mass_regression"],
        }
