import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
# Removed sklearn.metrics.jaccard_score, f1_score as PixelMetrics handles it
import time
from tqdm import tqdm
import json
import warnings
import pandas as pd
import seaborn as sns
import torch.cuda.amp as amp # NEW: Import for Automatic Mixed Precision (AMP)

warnings.filterwarnings("ignore")

# --- Import custom loss functions ---
try:
    from losses.combo import ComboLoss
    from losses.dice import DiceLoss
    from losses.focal import FocalLoss
    from losses.focal_tversky import FocalTverskyLoss
    from losses.jaccard import JaccardLoss
    from losses.lovasz_softmax import LovaszSoftmaxLoss
    from losses.topk import TopKLoss
    from losses.tversky import TverskyLoss
    from losses.dice_focal import CombinedLoss
except ImportError:
    print("Warning: Custom loss functions not found. Ensure 'losses' folder is in PYTHONPATH or same directory.")
    print("CrossEntropyLoss will be used as a fallback if a custom loss is requested and not found.")


# All dataset-related classes moved to dataset.py
# DeepGlobeDataset and PatchedDeepGlobeDataset are no longer here.

# --- NEW: PixelMetrics Class for Detailed Segmentation Metrics ---
class PixelMetrics:
    """
    A comprehensive class to track and compute pixel-level segmentation metrics,
    including overall and per-class IoU, F1-score, and Recall, using a confusion matrix.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # Confusion matrix: rows are true labels, columns are predicted labels
        self.conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.reset()

    def update(self, preds, targets):
        """
        Updates the confusion matrix with predictions and ground truth targets.
        Args:
            preds (torch.Tensor): Predicted masks (class indices). Shape (N, H, W).
            targets (torch.Tensor): Ground truth masks (class indices). Shape (N, H, W).
        """
        # Flatten predictions and targets for batch processing
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Ensure that target values are within the valid range [0, num_classes - 1]
        # and ignore any pixels with target values outside this range if they exist
        valid_pixels_mask = (targets_flat >= 0) & (targets_flat < self.num_classes)
        preds_flat = preds_flat[valid_pixels_mask]
        targets_flat = targets_flat[valid_pixels_mask]

        # Accumulate confusion matrix using torch.bincount
        # conf_matrix[i, j] is the count of pixels truly class i predicted as class j
        self.conf_matrix += torch.bincount(
            self.num_classes * targets_flat + preds_flat,
            minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes).cpu() # Accumulate on CPU to avoid device memory issues

    def reset(self):
        """Resets the confusion matrix to zero."""
        self.conf_matrix.fill_(0)

    def compute(self):
        """
        Calculates and returns a dictionary of overall and per-class metrics.
        Returns:
            dict: Contains 'overall_accuracy', 'mIoU', 'mF1', 'per_class_iou',
                  'per_class_f1', 'per_class_recall', and 'confusion_matrix'.
        """
        # Overall Pixel Accuracy
        total_correct_pixels = torch.diag(self.conf_matrix).sum().item()
        total_pixels = self.conf_matrix.sum().item()
        overall_accuracy = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0

        # Calculate True Positives (TP), False Positives (FP), False Negatives (FN) per class
        tp_per_class = torch.diag(self.conf_matrix).float()
        fp_per_class = self.conf_matrix.sum(0).float() - tp_per_class # Sum of column - diagonal
        fn_per_class = self.conf_matrix.sum(1).float() - tp_per_class # Sum of row - diagonal

        # Add a small epsilon for numerical stability to avoid division by zero
        epsilon = 1e-6

        # Per-class IoU (Jaccard Index): TP / (TP + FP + FN)
        iou_per_class = tp_per_class / (tp_per_class + fp_per_class + fn_per_class + epsilon)

        # Per-class Precision: TP / (TP + FP)
        precision_per_class = tp_per_class / (tp_per_class + fp_per_class + epsilon)

        # Per-class Recall (Sensitivity or True Positive Rate): TP / (TP + FN)
        # This is often what people mean by "accuracy for a specific class" in segmentation.
        recall_per_class = tp_per_class / (tp_per_class + fn_per_class + epsilon)

        # Per-class F1-score (Dice Coefficient): 2 * Precision * Recall / (Precision + Recall)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)

        # Mean IoU and Mean F1 (typically excludes background/unknown class for better representation)
        # For DeepGlobe, class 0 is often "Unknown" or "No-data".
        # Adjust `relevant_classes` based on your `num_classes` and class definitions.
        # If your class 0 is a valid class you want to include, change to slice(0, self.num_classes)
        # If your classes are 0-indexed and 0 is background, often mean is calculated for 1 to N-1 classes.
        relevant_classes = slice(1, self.num_classes) # Excludes class 0 (Unknown) for mean metrics

        # Ensure relevant_classes slice is valid if num_classes is small
        if relevant_classes.start >= self.num_classes:
            # No relevant classes to average, means will be NaN or 0
            mean_iou = torch.tensor(float('nan')).item()
            mean_f1 = torch.tensor(float('nan')).item()
        else:
            mean_iou = iou_per_class[relevant_classes].mean().item()
            mean_f1 = f1_per_class[relevant_classes].mean().item()

        results = {
            'overall_accuracy': overall_accuracy, # Overall pixel accuracy
            'mIoU': mean_iou,                   # Mean IoU over relevant classes
            'mF1': mean_f1,                     # Mean F1 over relevant classes
            'per_class_iou': iou_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'confusion_matrix': self.conf_matrix.tolist() # Can be large, useful for debugging
        }
        return results

    def compute_current(self):
        """Computes current metrics without resetting the confusion matrix."""
        # This is used for tqdm postfix, so it should be lightweight
        # Re-using the same logic as compute but without resetting
        return self.compute() # Calling compute is fine, it doesn't reset


# --- TrainingPipeline Class ---
class TrainingPipeline:
    """
    Encapsulates the training and validation loops for a segmentation model.
    Handles model training, evaluation, metric calculation, and logging.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_classes=7):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes

        # NEW: Single history dictionary to store all metrics (including per-class lists)
        self.history = {
            'train_loss': [], 'train_overall_accuracy': [], 'train_mIoU': [], 'train_mF1': [],
            'val_loss': [], 'val_overall_accuracy': [], 'val_mIoU': [], 'val_mF1': [],
            'val_per_class_iou': [], 'val_per_class_f1': [], 'val_per_class_recall': [] # Per-class metrics history
        }

        # NEW: Best validation mIoU tracking
        self.best_val_mIoU = -1.0
        # NEW: GradScaler for Automatic Mixed Precision (AMP)
        self.scaler = amp.GradScaler()

        # Clear CUDA cache at the start of pipeline init
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared at pipeline initialization.")


    # Removed calculate_metrics method, it's now handled by PixelMetrics

    def train(self, num_epochs, model_save_path):
        """
        Main training loop.
        Args:
            num_epochs (int): Number of epochs to train for.
            model_save_path (str): Path to save the best model.
        """
        print(f"Starting training for {num_epochs} epochs.")
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 80)

            start_time = time.time()

            # Train epoch
            train_metrics = self._train_epoch()
            # Validate epoch
            val_metrics = self._validate_epoch()

            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - start_time

            # Store metrics in history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_overall_accuracy'].append(train_metrics['overall_accuracy'])
            self.history['train_mIoU'].append(train_metrics['mIoU'])
            self.history['train_mF1'].append(train_metrics['mF1'])

            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_overall_accuracy'].append(val_metrics['overall_accuracy'])
            self.history['val_mIoU'].append(val_metrics['mIoU'])
            self.history['val_mF1'].append(val_metrics['mF1'])
            # Store per-class metrics lists
            self.history['val_per_class_iou'].append(val_metrics['per_class_iou'])
            self.history['val_per_class_f1'].append(val_metrics['per_class_f1'])
            self.history['val_per_class_recall'].append(val_metrics['per_class_recall'])


            print(f'Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["overall_accuracy"]:.4f}, IoU: {train_metrics["mIoU"]:.4f}, F1: {train_metrics["mF1"]:.4f}')
            print(f'Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["overall_accuracy"]:.4f}, IoU: {val_metrics["mIoU"]:.4f}, F1: {val_metrics["mF1"]:.4f}')
            print(f'Time: {epoch_time:.2f}s')

            # Save best model based on validation mIoU
            if val_metrics['mIoU'] > self.best_val_mIoU:
                self.best_val_mIoU = val_metrics['mIoU']
                torch.save(self.model.state_dict(), model_save_path)
                print(f'Model saved to {model_save_path} with best Val mIoU: {self.best_val_mIoU:.4f}')
        print("\nTraining complete.")

    def _train_epoch(self):
        """Performs one training epoch."""
        self.model.train()
        running_loss = 0.0
        # Use PixelMetrics instead of MetricsTracker
        metric_tracker = PixelMetrics(self.num_classes)
        pbar = tqdm(self.train_loader, desc='Training')

        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # --- AMP specific changes START ---
            with amp.autocast(): # Perform operations in mixed precision
                outputs = self.model(images)
                # Resize targets to match output spatial dimensions if necessary
                targets_resized = nn.functional.interpolate(
                    targets.unsqueeze(1).float(),
                    size=outputs.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()
                loss = self.criterion(outputs, targets_resized)

            self.scaler.scale(loss).backward() # Scale loss and perform backward pass
            self.scaler.step(self.optimizer)   # Update optimizer weights
            self.scaler.update()               # Update the scaler for the next iteration
            # --- AMP specific changes END ---

            running_loss += loss.item() * images.size(0)

            # Calculate metrics using PixelMetrics
            predicted_masks = outputs.argmax(1)
            metric_tracker.update(predicted_masks, targets_resized)

            # Update progress bar with overall metrics
            current_metrics = metric_tracker.compute_current()
            pbar.set_postfix(
                {'Loss': f'{loss.item():.4f}',
                 'Acc': f'{current_metrics.get("overall_accuracy", 0):.4f}',
                 'mIoU': f'{current_metrics.get("mIoU", 0):.4f}',
                 'mF1': f'{current_metrics.get("mF1", 0):.4f}'
                }
            )

        avg_loss = running_loss / len(self.train_loader.dataset)
        avg_metrics = metric_tracker.compute() # This will contain all per-class metrics
        avg_metrics['loss'] = avg_loss # Add loss to the metrics dictionary
        return avg_metrics


    def _validate_epoch(self):
        """Performs one validation epoch."""
        self.model.eval()
        running_loss = 0.0
        # Use PixelMetrics instead of MetricsTracker
        metric_tracker = PixelMetrics(self.num_classes)
        pbar = tqdm(self.val_loader, desc='Validation')

        with torch.no_grad():
            with amp.autocast(): # Use autocast for evaluation as well
                for images, targets in pbar:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                    targets_resized = nn.functional.interpolate(
                        targets.unsqueeze(1).float(),
                        size=outputs.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                    loss = self.criterion(outputs, targets_resized)
                    running_loss += loss.item() * images.size(0)

                    # Calculate metrics using PixelMetrics
                    predicted_masks = outputs.argmax(1)
                    metric_tracker.update(predicted_masks, targets_resized)

                    # Update progress bar
                    current_metrics = metric_tracker.compute_current()
                    pbar.set_postfix(
                        {'Loss': f'{loss.item():.4f}',
                         'Acc': f'{current_metrics.get("overall_accuracy", 0):.4f}',
                         'mIoU': f'{current_metrics.get("mIoU", 0):.4f}',
                         'mF1': f'{current_metrics.get("mF1", 0):.4f}'
                        }
                    )

        avg_loss = running_loss / len(self.val_loader.dataset)
        avg_metrics = metric_tracker.compute() # This will contain all per-class metrics
        avg_metrics['loss'] = avg_loss # Add loss to the metrics dictionary
        return avg_metrics


    def evaluate(self, data_loader):
        """
        Evaluates the model on a given DataLoader.
        Typically used for validation or final test sets.
        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader for evaluation.
        Returns:
            dict: A dictionary of metrics including overall and per-class details.
        """
        self.model.eval() # Set model to evaluation mode
        running_loss = 0.0
        metric_tracker = PixelMetrics(self.num_classes)

        with torch.no_grad(): # Disable gradient calculations for evaluation
            with amp.autocast(): # Use autocast here as well for evaluation
                for images, masks in tqdm(data_loader, desc="Evaluating"): # Added tqdm for eval
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)

                    targets_resized = nn.functional.interpolate(
                        masks.unsqueeze(1).float(),
                        size=outputs.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()

                    loss = self.criterion(outputs, targets_resized)
                    running_loss += loss.item() * images.size(0)

                    predicted_masks = outputs.argmax(1) # Get predicted class for each pixel
                    metric_tracker.update(predicted_masks, targets_resized)

        avg_loss = running_loss / len(data_loader.dataset)
        # Compute all metrics, including per-class ones
        final_metrics = metric_tracker.compute()
        final_metrics['loss'] = avg_loss # Add loss to the metrics dictionary

        return final_metrics

    def plot_metrics(self, save_path):
        """Save individual metric plots as separate files."""
        # Update metric names to match the new history keys
        metrics = [
            ('loss', 'Loss'),
            ('overall_accuracy', 'Overall Accuracy'),
            ('mIoU', 'Mean IoU'),
            ('mF1', 'Mean F1 Score')
        ]

        plt.style.use('default') # Ensure default style for plots

        for metric_key, title_name in metrics:
            plt.figure(figsize=(10, 6))
            # Use the correct history keys
            plt.plot(self.history[f'train_{metric_key}'], label='Train', marker='o')
            plt.plot(self.history[f'val_{metric_key}'], label='Validation', marker='s')
            plt.title(f'{title_name} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(title_name)
            plt.legend()
            plt.grid(True, alpha=0.3)

            metric_path = os.path.join(save_path, f'{metric_key}_plot.png')
            plt.savefig(metric_path, dpi=300, bbox_inches='tight')
            plt.close()
        print(f"Metric plots saved to {save_path}")

# --- Utility functions (denormalize, colorize_mask, visualize_predictions, get_loss_function, create_transforms_from_config, create_comparison_plots) remain mostly the same ---

def denormalize(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    # Create copies to avoid modifying the original tensor
    mean_tensor = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)

    # If the input tensor has a batch dimension, reshape mean/std for broadcasting
    if tensor.dim() == 4:
        # For batch of images (N, C, H, W)
        denorm_tensor = tensor * std_tensor + mean_tensor
    elif tensor.dim() == 3:
        # For single image (C, H, W)
        denorm_tensor = tensor * std_tensor.squeeze(0) + mean_tensor.squeeze(0)
    else:
        raise ValueError("Input tensor must be 3D (C, H, W) or 4D (N, C, H, W).")

    return denorm_tensor


def colorize_mask(mask, colormap):
    """Converts a segmentation mask (class indices) to a color image."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(colormap):
        color_mask[mask == class_idx] = color
    return color_mask


def visualize_predictions(model, dataset, device, config, save_path):
    """Shows predictions and saves individual visualization images."""
    num_samples = config.get('visualize_samples', 0)
    if num_samples == 0:
        print("Skipping visualization as 'visualize_samples' is set to 0.")
        return

    print(f"\nGenerating {num_samples} prediction visualizations...")
    model.eval()

    colormap = [
        (0, 0, 0),             # 0=Unknown
        (0, 255, 255),         # 1=Urban
        (255, 255, 0),         # 2=Agriculture
        (255, 0, 255),         # 3=Rangeland
        (0, 255, 0),           # 4=Forest
        (0, 0, 255),           # 5=Water
        (255, 255, 255)        # 6=Barren
    ]

    indices = np.random.choice(range(len(dataset)), min(num_samples, len(dataset)), replace=False)

    viz_dir = os.path.join(save_path, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    with torch.no_grad():
        with amp.autocast(): # Add autocast for visualization
            for i, idx in enumerate(indices):
                image, gt_mask = dataset[idx]
                image_for_model = image.unsqueeze(0).to(device)
                output = model(image_for_model)
                # Ensure output is on CPU before numpy conversion for large tensors
                pred_mask_resized = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                original_image = denormalize(image.clone(), **config['normalization'])
                original_image = original_image.permute(1, 2, 0).cpu().numpy()
                original_image = np.clip(original_image, 0, 1)

                gt_mask_np = gt_mask.cpu().numpy()
                gt_colored = colorize_mask(gt_mask_np, colormap)
                pred_colored = colorize_mask(pred_mask_resized, colormap)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(original_image)
                axes[0].set_title(f'Original Image #{idx}')
                axes[0].axis('off')

                axes[1].imshow(gt_colored)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred_colored)
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                plt.tight_layout()
                viz_path = os.path.join(viz_dir, f'prediction_{i+1}_{idx}.png')
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()


def get_loss_function(loss_name):
    """
    Returns the specified loss function.
    Args:
        loss_name (str): Name of the loss function (e.g., 'cross_entropy', 'dice', 'focal').
    Returns:
        torch.nn.Module: The instantiated loss function.
    """
    loss_functions = {
        'cross_entropy': nn.CrossEntropyLoss(),
    }
    try:
        if 'ComboLoss' in globals(): loss_functions['combo'] = ComboLoss()
        if 'DiceLoss' in globals(): loss_functions['dice'] = DiceLoss()
        if 'FocalLoss' in globals(): loss_functions['focal'] = FocalLoss()
        if 'FocalTverskyLoss' in globals(): loss_functions['focal_tversky'] = FocalTverskyLoss()
        if 'JaccardLoss' in globals(): loss_functions['jaccard'] = JaccardLoss()
        if 'LovaszSoftmaxLoss' in globals(): loss_functions['lovasz'] = LovaszSoftmaxLoss()
        if 'TopKLoss' in globals(): loss_functions['topk'] = TopKLoss()
        if 'TverskyLoss' in globals(): loss_functions['tversky'] = TverskyLoss()
        if 'CombinedLoss' in globals(): loss_functions['dice_focal'] = TverskyLoss()
    except NameError:
        pass

    if loss_name.lower() not in loss_functions:
        print(f"Warning: Loss function '{loss_name}' not found. Falling back to CrossEntropyLoss.")
        return nn.CrossEntropyLoss()
    return loss_functions[loss_name.lower()]


def create_transforms_from_config(config):
    """
    Creates image and mask transformations based on the provided configuration.
    Args:
        config (dict): Configuration dictionary containing 'image_size', 'augmentation', 'normalization'.
    Returns:
        tuple: (train_transform, val_transform, target_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(p=config['augmentation']['horizontal_flip']),
        transforms.RandomVerticalFlip(p=config['augmentation']['vertical_flip']),
        transforms.ColorJitter(**config['augmentation']['color_jitter']),
        transforms.ToTensor(),
        transforms.Normalize(**config['normalization'])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(**config['normalization'])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=Image.NEAREST),
    ])

    return train_transform, val_transform, target_transform


def create_comparison_plots(results_df, master_experiment_path, focus_hyperparameter):
    """
    Creates and saves bar plots comparing different metrics across multiple experiment runs.
    Args:
        results_df (pd.DataFrame): DataFrame containing results from all runs.
        master_experiment_path (str): Base directory to save the comparison plots.
        focus_hyperparameter (str): The hyperparameter to highlight/color bars by in plots.
    """
    # Updated metrics list to match new names
    metrics = [
        'train_loss', 'val_loss',
        'train_overall_accuracy', 'val_overall_accuracy',
        'train_mIoU', 'val_mIoU',
        'train_mF1', 'val_mF1',
        'final_test_loss', 'final_test_overall_accuracy', 'final_test_mIoU', 'final_test_mF1' # Added test metrics
    ]

    plt.style.use('default')
    sns.set_palette("husl")

    for metric in metrics:
        if metric not in results_df.columns:
            print(f"Skipping plot for {metric}: column not found in results.")
            continue

        plt.figure(figsize=(12, 6))

        bars = plt.bar(range(len(results_df)), results_df[metric], alpha=0.7)

        if focus_hyperparameter in results_df.columns:
            unique_values = results_df[focus_hyperparameter].unique()
            colors = sns.color_palette("husl", len(unique_values))
            color_map = dict(zip(unique_values, colors))

            for i, bar in enumerate(bars):
                if pd.notna(results_df.iloc[i][focus_hyperparameter]):
                    bar.set_color(color_map[results_df.iloc[i][focus_hyperparameter]])

        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Runs')
        plt.xlabel('Run')
        plt.ylabel(metric.replace("_", " ").title())

        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        if focus_hyperparameter in results_df.columns and len(unique_values) > 1:
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[val], alpha=0.7)
                               for val in unique_values if pd.notna(val)]
            plt.legend(legend_elements, [str(val) for val in unique_values if pd.notna(val)],
                       title=focus_hyperparameter.replace('_', ' ').title(),
                       bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.8)

        labels = [f"Run {i+1}" for i in range(len(results_df))]
        plt.xticks(range(len(results_df)), labels, rotation=45, ha='right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(master_experiment_path, f'{metric}_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Comparison plots saved to {master_experiment_path}")