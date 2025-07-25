import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import os
from sklearn.metrics import roc_curve, auc
import pandas as pd
from dataset import JetToCalorimeterDataset, get_dataloader
from model import load_pretrained_model
from jet_utils import load_images
verbose = True  # Set to False to disable detailed output

def load_processed_data():
    """
    Load processed data with unique IDs and tabular features.
    """
    import pandas as pd
    
    # Load training data
    X_train = pd.read_csv('./qcd-tt-jet-tagging-co-da-s-hep/train/features/cluster_features.csv')
    y_train = np.load('./qcd-tt-jet-tagging-co-da-s-hep/train/labels/labels.npy')
    train_ids = np.load('./qcd-tt-jet-tagging-co-da-s-hep/train/ids/ids.npy')

    # Load validation data
    X_val = pd.read_csv('./qcd-tt-jet-tagging-co-da-s-hep/val/features/cluster_features.csv')
    y_val = np.load('./qcd-tt-jet-tagging-co-da-s-hep/val/labels/labels.npy')
    val_ids = np.load('./qcd-tt-jet-tagging-co-da-s-hep/val/ids/ids.npy')
    
    # Load test data
    X_test = pd.read_csv('./qcd-tt-jet-tagging-co-da-s-hep/test/features/cluster_features.csv')
    test_ids = np.load('./qcd-tt-jet-tagging-co-da-s-hep/test/ids/ids.npy')
    
    return X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids


def train_epoch(model, train_loader, optimizer, criterion, device, alpha=1.0, beta=0.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    regression_criterion = nn.MSELoss()
    
    for batch_idx, (sparse_tensor, labels, regression_targets) in enumerate(train_loader):
        # Forward pass
        optimizer.zero_grad()
        
        # Model outputs both classification and regression
        classification_outputs, regression_outputs = model(sparse_tensor)
        
        # Reshape for loss calculation
        labels = labels.float().view(-1, 1)
        regression_targets = regression_targets.float().view(-1, 1)
        
        # Classification loss
        class_loss = criterion(classification_outputs, labels)
        
        # Regression loss (always calculated, weighted by beta)
        reg_loss = regression_criterion(regression_outputs, regression_targets)
        
        # Total loss with weighting
        loss = alpha * class_loss + beta * reg_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predicted = (classification_outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f} (Class: {class_loss.item():.4f}, Reg: {reg_loss.item():.4f})')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device, alpha=1.0, beta=0.0):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_scores = []
    all_labels = []
    
    regression_criterion = nn.MSELoss()
    
    with torch.no_grad():
        for sparse_tensor, labels, regression_targets in val_loader:
            # Forward pass
            classification_outputs, regression_outputs = model(sparse_tensor)
            
            # Reshape
            labels = labels.float().view(-1, 1)
            regression_targets = regression_targets.float().view(-1, 1)
            
            # Losses
            class_loss = criterion(classification_outputs, labels)
            reg_loss = regression_criterion(regression_outputs, regression_targets)
            loss = alpha * class_loss + beta * reg_loss
            
            # Statistics
            total_loss += loss.item()
            predicted = (classification_outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Store for AUC calculation
            all_scores.extend(classification_outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    # Calculate AUC
    if len(set(all_labels)) > 1:  # Need both classes for AUC
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0.0
    
    return avg_loss, accuracy, roc_auc


def main():
    val_auc_scores = []
    parser = argparse.ArgumentParser(description='Cross-train calorimeter CNN on jet data')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--model-path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='./jet_training_output', help='Output directory')
    parser.add_argument('--layer-idx', type=int, default=1, help='Which calorimeter layer to place jet data (0,1,2)')
    parser.add_argument('--lr-patience', type=int, default=5, help='Epochs to wait before reducing LR')
    parser.add_argument('--lr-factor', type=float, default=0.5, help='Factor to reduce LR by')
    parser.add_argument('--min-lr', type=float, default=1e-8, help='Minimum learning rate')
    parser.add_argument('--target-feature', type=str, default=None, 
                       help='CSV column to use as regression target (e.g., total_pt, max_cluster_pt)')
    parser.add_argument('--beta', type=float, default=0.0, 
                       help='Weight for regression loss (0.0 = classification only)')
    parser.add_argument('--alpha', type=float, default=1.0, 
                       help='Weight for classification loss (0.0 = classification only)')
    
    args = parser.parse_args()
    ENSEMBLE_FEATURES = [
        'total_pt', 'max_cluster_pt', 'cluster_pt_ratio', 
        'n_clusters', 'max_cluster_eta', 'mean_cluster_pt',
        'std_cluster_pt'  # Add more if you want
    ]

    if args.target_feature == 'ENSEMBLE':
        print("🚀 ENSEMBLE MODE: Training multiple models with different regression targets")
        ensemble_results = []
        
        for i, feature in enumerate(ENSEMBLE_FEATURES):
            print(f"\n{'='*60}")
            print(f"Training Model {i+1}/{len(ENSEMBLE_FEATURES)}: {feature}")
            print(f"{'='*60}")
            
            # Update args for this model
            args.target_feature = feature
            args.output_dir = f'./ensemble_models/model_{feature}'
            
            # Run the training (rest of your main() code runs here)
            # We'll return to normal flow, but capture results
            
            # Store this before we start training
            original_verbose = verbose
    
    # Print available features if none specified
    if args.target_feature is None and args.beta > 0:
        print("Warning: --beta > 0 but no --target-feature specified. Using dummy regression targets.")
        print("Available features: n_clusters, max_cluster_pt, mean_cluster_pt, std_cluster_pt,")
        print("                   max_cluster_size, mean_cluster_size, std_cluster_size, total_pt,")
        print("                   max_cluster_eta, max_cluster_phi, mean_cluster_eta, mean_cluster_phi,")
        print("                   cluster_pt_ratio, cluster_size_ratio")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load jet image data
    print("Loading jet image data...")
    X_train_images, y_train, train_ids, X_val_images, y_val, val_ids, X_test_images, test_ids = load_images()
    
    # Load tabular features if using regression
    X_train_features, X_val_features, X_test_features = None, None, None
    if args.target_feature is not None:
        print("Loading tabular features...")
        X_train_features, y_train_tab, train_ids_tab, X_val_features, y_val_tab, val_ids_tab, X_test_features, test_ids_tab = load_processed_data()
        
        # Verify data alignment
        if not np.array_equal(y_train, y_train_tab) or not np.array_equal(y_val, y_val_tab):
            print("Warning: Labels from images and tabular data don't match!")
        if not np.array_equal(train_ids, train_ids_tab) or not np.array_equal(val_ids, val_ids_tab):
            print("Warning: IDs from images and tabular data don't match!")
        
        print(f"Tabular features shape: {X_train_features.shape}")
        print(f"Available features: {list(X_train_features.columns)}")
        if args.target_feature in X_train_features.columns:
            feature_stats = X_train_features[args.target_feature].describe()
            print(f"Target feature '{args.target_feature}' statistics:")
            print(f"  {feature_stats}")
    
    print(f"Data loaded:")
    print(f"  Train: {len(X_train_images)} images, shape: {X_train_images[0].shape}")
    print(f"  Val: {len(X_val_images)} images")
    print(f"  Test: {len(X_test_images)} images")
    
    # Create datasets (convert jet images to sparse calorimeter format)
    print(f"Converting to sparse calorimeter format (layer {args.layer_idx})...")
    if args.beta > 0:
        print(f"Using regression target: {args.target_feature}, weight: {args.beta}")
    else:
        print("Classification only (beta = 0)")
        
    train_dataset = JetToCalorimeterDataset(
        X_train_images, y_train, 
        tabular_features=X_train_features, 
        target_feature=args.target_feature,
        layer_idx=args.layer_idx
    )
    val_dataset = JetToCalorimeterDataset(
        X_val_images, y_val,
        tabular_features=X_val_features,
        target_feature=args.target_feature, 
        layer_idx=args.layer_idx
    )
    
    # Create dataloaders
    train_loader = get_dataloader(train_dataset, args.batch_size, device, shuffle=True)
    val_loader = get_dataloader(val_dataset, args.batch_size, device, shuffle=False)
    
    print(f"Created dataloaders with batch size {args.batch_size}")
    
    # Load pretrained model
    print("Loading pretrained calorimeter model...")
    model = load_pretrained_model(args.model_path, device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler - reduces LR when validation metric plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',              # We want to maximize AUC
        factor=args.lr_factor,   # Reduce LR by this factor
        patience=args.lr_patience,  # Wait N epochs before reducing
        min_lr=args.min_lr,      # Don't go below this LR
        verbose=True             # Print when LR changes
    )
    
    # Training loop
    print(f"\nStarting cross-training for {args.epochs} epochs...")
    print(f"Initial learning rate: {args.lr}")
    print(f"Loss weights: α={1.0} (classification), β={args.beta} (regression)")
    if args.target_feature:
        print(f"Regression target: {args.target_feature}")
    print(f"LR Scheduler: Reduce by {args.lr_factor}x after {args.lr_patience} epochs without improvement")
    print(f"Minimum LR: {args.min_lr}")
    
    best_val_auc = 0.0
    best_model_state = None
    
    # Initialize training history for CSV
    training_history = []
    csv_path = os.path.join(args.output_dir, 'training_history.csv')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, alpha=args.alpha, beta=args.beta
        )
        
        # Validate  
        val_loss, val_acc, val_auc = validate_epoch(
            model, val_loader, criterion, device, alpha=args.alpha, beta=args.beta
        )
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        
        # Store epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'epoch_time': epoch_time,
            'learning_rate': current_lr,  # Use actual current LR
            'alpha': args.alpha,
            'beta': args.beta,
            'target_feature': args.target_feature if args.target_feature else 'none',
            'is_best': False
        }
        
        # Print results
        if(verbose):print(f"Epoch {epoch+1} Results ({epoch_time:.1f}s):")
        if(verbose):print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if(verbose): print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
        val_auc_scores.append(val_auc)
        if(verbose):print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            epoch_results['is_best'] = True
        
        # Update learning rate scheduler based on validation AUC
        scheduler.step(val_auc)
        
        # Check if learning rate was reduced
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr and verbose:
            print(f"  📉 Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
        
        # Check for overfitting warning
        if epoch > 5 and val_auc < max([h['val_auc'] for h in training_history[-5:]]) and verbose:
            print(f"  ⚠️  Warning: Val AUC hasn't improved in recent epochs (possible overfitting)")
        
        # Add to history and save CSV
        training_history.append(epoch_results)
        
        # Save/update CSV after each epoch
        df = pd.DataFrame(training_history)
        df.to_csv(csv_path, index=False)
        print(f"  Training progress saved to: {csv_path}")
    
    # Replace the existing final results section with this:
    
    # Calculate final metrics
    final_auc = np.max(val_auc_scores)
    final_results = {
        'feature': args.target_feature,
        'best_auc': final_auc,
        'beta': args.beta,
        'epochs': args.epochs,
        'lr': args.lr
    }
    
    # Save predictions for ensemble
    pred_path = os.path.join(args.output_dir, 'val_predictions.npy')
    
    # Get validation predictions from best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    val_preds = []
    with torch.no_grad():
        for sparse_tensor, _, _ in val_loader:
            outputs, _ = model(sparse_tensor)
            val_preds.extend(outputs.cpu().numpy().flatten())
    
    np.save(pred_path, val_preds)
    print(f"Validation predictions saved to: {pred_path}")
    
    # If ensemble mode, collect results
    if args.target_feature in ENSEMBLE_FEATURES:
        ensemble_results.append(final_results)
    
    auc = "AUC: " + str(final_auc)
    beta = "Beta: " + str(args.beta) 
    feature = "Feature: " + str(args.target_feature)
    print(auc + "," + beta + "," + feature)
    
    # If we're in ensemble mode and this is the last model
    if args.target_feature in ENSEMBLE_FEATURES and len(ensemble_results) == len(ENSEMBLE_FEATURES):
        # Save ensemble summary
        ensemble_df = pd.DataFrame(ensemble_results)
        ensemble_df.to_csv('./ensemble_summary.csv', index=False)
        print("\n🎯 ENSEMBLE TRAINING COMPLETE!")
        print(f"Summary saved to: ./ensemble_summary.csv")
        return ensemble_results
    
    return final_results



if __name__ == '__main__':
    main()