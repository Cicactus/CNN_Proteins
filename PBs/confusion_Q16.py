import torch
import torch.nn as nn
import argparse
import numpy as np
import csv
# Removed get_loader_accuracy_CNN as it's no longer needed for accuracy
from CNN_Q16 import activations, create_dataloader
from sklearn.metrics import classification_report
import collections
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

parser = argparse.ArgumentParser(description="argparse")
parser.add_argument("--model_path", type=str, default='./Q16_BEST_MODEL_7_5_3_2_1_512_78.52_FOR_CPU')
parser.add_argument("--data_path", type=str, default='./alldb/CB513X.npy')
parser.add_argument("--conv_activation", type=str, default='tanh')
parser.add_argument("--fc_activation", type=str, default='leakyrelu')
args = parser.parse_args()

device = torch.device("cpu")

# Model Architecture Parsing
match = re.search(r'_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_FOR_CPU$', args.model_path)
if match:
    # Note: match.group(1) is conv_layers, (2) is fc_layers, etc.
    conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels = map(int, match.groups()[:6])
    # The last group is accuracy, which we don't need for model def
else:
    # Fallback or raise an error if parsing fails for robustness
    print("Warning: Could not parse model architecture from filename. Using default values.")
    conv_layers = 2
    fc_layers = 1
    kernel_size = 10
    stride = 2
    padding = 1
    first_layer_channels = 32

n_outputs = 16  # Q16

# Model definition (no changes needed here, it outputs logits)
class Model(nn.Module):
    def __init__(self, freeze_conv=False):
        super().__init__()

        # Input length (380) should ideally come from X.shape[-1] if possible,
        # but 380 is a common fixed input length for Q16.
        l = 380
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.conv_activation = activations[args.conv_activation]
        self.fc_activation = activations[args.fc_activation]

        for i in range(conv_layers):
            ch = first_layer_channels
            if i == 0:
                conv = nn.Conv1d(1, ch, kernel_size, stride, padding)
            else:
                conv = nn.Conv1d(ch, ch, kernel_size, stride, padding)
            bn = nn.BatchNorm1d(ch)
            self.convs.append(conv)
            self.bns.append(bn)
            l = int((l + 2 * padding - kernel_size) / stride + 1)

        l *= ch
        if l <= 1:
            raise ValueError(f"Invalid input length after convolution layers: {l}")

        powers = np.linspace(
            int(np.floor(np.log2(l - 1))),
            int(np.ceil(np.log2(n_outputs + 1))),
            fc_layers - 1
        )
        values = [l] + [int(2 ** p) for p in powers] + [n_outputs]

        for i in range(fc_layers):
            fc = nn.Linear(values[i], values[i + 1])
            self.fcs.append(fc)

        if freeze_conv:
            for conv in self.convs:
                for param in conv.parameters():
                    param.requires_grad = False
            for bn in self.bns:
                for param in bn.parameters():
                    param.requires_grad = False

    def forward(self, x):
        for i in range(conv_layers):
            x = self.conv_activation(self.bns[i](self.convs[i](x)))
        x = x.flatten(start_dim=1)
        for i in range(len(self.fcs) - 1):
            x = self.fc_activation(self.fcs[i](x))
        return self.fcs[-1](x) 

def load_model(model_path):
    model = Model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Load model and data
model = load_model(args.model_path).to(device)

X = np.load(args.data_path).astype('float32')
X = X.reshape(X.shape[0], 1, X.shape[-1])

Y_path = args.data_path.replace('X.npy', 'Y.npy') # More robust way to derive Y_path
Yt_rmsd_path = 'yt.csv'

has_labels = os.path.exists(Y_path)
has_rmsd = not has_labels and os.path.exists(Yt_rmsd_path) # Check yt.csv only if Y.npy is missing

y_true_rmsd_data = None # Store the actual RMSD array if available

if has_labels:
    y_true_rmsd_data = np.load(Y_path).astype('float32')
    # For dataloader, we pass the RMSD array.
    loader = create_dataloader(X, y_true_rmsd_data, 1024)

elif has_rmsd:
    print("Y.npy not found, using yt.csv to generate labels based on RMSD (argmin).")
    rmsd = np.loadtxt(Yt_rmsd_path, delimiter='\t').astype('float32') # Ensure float32 for consistency
    y_true_rmsd_data = rmsd # Store the RMSD data directly
    # Create dataloader with RMSD data
    loader = create_dataloader(X, y_true_rmsd_data, 1024)
    has_labels = True # Now we have effective labels from RMSD

else:
    y_true_rmsd_data = None
    print("No labels found — only generating predictions.")
    # If no labels, create a DataLoader with dummy Y for consistency in loop structure
    from torch.utils.data import TensorDataset, DataLoader # Ensure these are available if not imported from general_functions
    dummy_y = np.zeros((X.shape[0], n_outputs), dtype='float32')
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(dummy_y)), batch_size=1024, shuffle=False)


y_pred_logits = [] # Store raw logits from model
y_true_rmsd_batches = [] # Store RMSD targets as they come from loader

print("Running prediction...")
with torch.no_grad(): # Essential for inference to save memory and computations
    model.eval() # Set model to evaluation mode
    for i, (Xt_batch, yt_batch_rmsd) in enumerate(loader):
        Xt_batch = Xt_batch.to(device)
        preds_logits = model(Xt_batch).detach().cpu().numpy() # Model outputs logits
        y_pred_logits.append(preds_logits)
        y_true_rmsd_batches.append(yt_batch_rmsd.numpy()) # Keep original RMSD values

        if (i + 1) % 10 == 0:
            print(f"Processed {((i + 1) * loader.batch_size):,} samples")

y_pred_logits = np.vstack(y_pred_logits)
# Only combine true RMSD data if labels were actually available
if y_true_rmsd_data is not None:
    y_true_rmsd_data_combined = np.vstack(y_true_rmsd_batches)
else:
    y_true_rmsd_data_combined = None


# Save predictions (these are now raw logits)
print("Saving predictions (raw logits)...")
with open("y_pred_logits.csv", 'w', newline='') as f: # Renamed to reflect it's logits
    writer = csv.writer(f)
    writer.writerows(y_pred_logits)
print("y_pred_logits.csv saved")

# --- Compute metrics if labels are available ---
if y_true_rmsd_data_combined is not None: # Check if true labels were available for the dataset
    print("\n" + "-"*30)
    print("Calculating Classification Metrics")
    print("-" * 30)

    # For predictions, use argmax on logits to get the most likely class.
    y_pred_indices = np.argmax(y_pred_logits, axis=1)
    # For true labels, use argmin on RMSD values to get the true class.
    y_true_indices = np.argmin(y_true_rmsd_data_combined, axis=1)

    # --- Confusion Matrix ---
    print("\nCalculating confusion matrix...")
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=np.arange(n_outputs))
    cm_normalized = cm.astype(np.float32)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Avoid division by zero for classes with no true samples
    cm_normalized = cm_normalized / row_sums * 100

    print("Confusion matrix (normalized %, sklearn):")
    print(cm_normalized)
    np.savetxt("confusion_matrix.csv", cm_normalized, fmt='%7.3f', delimiter='\t')
    print("confusion_matrix.csv saved")

    # --- Heatmap ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap='Blues',
                xticklabels=np.arange(n_outputs), yticklabels=np.arange(n_outputs))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix (%)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_plot.png")
    plt.show()

    # --- Distribution of True Classes ---
    class_counts = collections.Counter(y_true_indices)
    print("\nРаспределение классов (реальные данные):")
    for label in sorted(class_counts):
        print(f"Класс {label:2d}: {class_counts[label]} примеров")

    # --- Classification Report ---
    print("\nКлассификационный отчёт:")
    report = classification_report(
        y_true_indices,
        y_pred_indices,
        labels=np.arange(n_outputs),
        digits=3,
        output_dict=True,
        zero_division='warn'
    )

    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")

    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("classification_report.json saved.")

else:
    print("\nNo true labels were found, so confusion matrix and classification report cannot be generated.")