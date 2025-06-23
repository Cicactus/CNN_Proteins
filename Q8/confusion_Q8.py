import torch
import torch.nn as nn
import argparse
import numpy as np
import csv
from CNN_Q8 import activations, create_dataloader
from sklearn.metrics import classification_report
import collections
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path # Import Path for easier path manipulation

parser = argparse.ArgumentParser(description="argparse")
parser.add_argument("--model_path", type=str, default='Q8_BEST_MODEL_7_5_3_2_1_512_tanh_leakyrelu_68.48_FOR_CPU')
parser.add_argument("--data_path", type=str, default='./learn_Q8/test_for_dsspX.npy')
args = parser.parse_args()

device = torch.device("cpu")

# Function to extract model parameters from filename (similar to the inference script)
def extract_model_params_from_filename(model_path):
    match = re.search(r"_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_([a-zA-Z]+)_([a-zA-Z]+)_(\d+\.\d+)", Path(model_path).name)
    if match:
        conv_layers = int(match.group(1))
        fc_layers = int(match.group(2))
        kernel_size = int(match.group(3))
        stride = int(match.group(4))
        padding = int(match.group(5))
        first_layer_channels = int(match.group(6))
        conv_activation_name = match.group(7)
        fc_activation_name = match.group(8)
        return conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels, conv_activation_name, fc_activation_name
    return None, None, None, None, None, None, None, None

# Initialize with default values, which will be overridden if parsing is successful
conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels = 2, 1, 10, 2, 1, 32
conv_activation = 'tanh'
fc_activation = 'leakyrelu'

# Attempt to extract parameters from the filename
parsed_params = extract_model_params_from_filename(args.model_path)
if all(p is not None for p in parsed_params):
    conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels, conv_activation, fc_activation = parsed_params
    print("Parsed model parameters from filename successfully.")
else:
    print("Warning: Could not parse all architecture parameters or activation functions from model filename. "
          "Using hardcoded defaults. Ensure filename matches expected pattern (e.g., Q8_BEST_MODEL_..._2_1_10_2_1_32_tanh_leakyrelu_...).")

n_outputs = 8  # Q8

# Model definition
class Model(nn.Module):
    def __init__(self, input_len, freeze_conv=False): # Added input_len to __init__
        super().__init__()

        l = input_len # Use input_len from parameter
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.conv_activation = activations[conv_activation] # Use parsed activation
        self.fc_activation = activations[fc_activation]     # Use parsed activation

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

        flat_dim = l * ch
        if flat_dim <= 1:
            raise ValueError(f"Invalid input length after convolution layers: {flat_dim}")

        powers = np.linspace(
            int(np.floor(np.log2(flat_dim - 1))),
            int(np.ceil(np.log2(n_outputs + 1))),
            fc_layers - 1
        )
        values = [flat_dim] + [int(2 ** p) for p in powers] + [n_outputs]

        for i in range(fc_layers):
            fc = nn.Linear(values[i], values[i + 1])
            self.fcs.append(fc)

        if freeze_conv:
            for conv_layer in self.convs: # Renamed for clarity
                for param in conv_layer.parameters():
                    param.requires_grad = False
            for bn_layer in self.bns: # Renamed for clarity
                for param in bn_layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        for i in range(conv_layers):
            x = self.conv_activation(self.bns[i](self.convs[i](x)))
        x = x.flatten(start_dim=1)
        for i in range(len(self.fcs) - 1):
            x = self.fc_activation(self.fcs[i](x))
        return self.fcs[-1](x) # This correctly outputs raw logits

def load_model_and_data(model_path, data_path):
    # Load X data first to determine input_len for the model
    X = np.load(data_path).astype('float32')
    X = X.reshape(X.shape[0], 1, X.shape[-1]) # Reshape to (batch, channels, length)
    input_len_from_data = X.shape[-1]

    # Initialize model using the parsed parameters and detected input_len
    model = Model(input_len=input_len_from_data)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Derive Y_path from data_path
    base_name = Path(data_path).stem.replace('X', 'Y')
    Y_path = Path(data_path).parent / f"{base_name}.npy"
    
    y_true_data = None
    if Y_path.exists():
        y_true_data = np.load(Y_path).astype('float32')
        loader = create_dataloader(X, y_true_data, 1024)
        print(f"Loaded labels from: {Y_path}")
    else:
        print(f"Warning: Label file not found for {data_path} at {Y_path}. Proceeding without labels for metrics.")
        # Create a dummy Y for the DataLoader if no labels are found, to ensure loop consistency
        dummy_y = np.zeros((X.shape[0], n_outputs), dtype='float32')
        from torch.utils.data import TensorDataset, DataLoader
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(dummy_y)), batch_size=1024, shuffle=False)

    return model, loader, y_true_data # Return y_true_data as it might be None

# Load model and data
model, loader, y_true_data_for_metrics = load_model_and_data(args.model_path, args.data_path)

y_pred_logits = [] # Store raw logits from model
y_true_batches = [] # Store true labels (one-hot or indices)

print("Running prediction...")
with torch.no_grad():
    model.eval()
    for i, (Xt_batch, yt_batch) in enumerate(loader):
        Xt_batch = Xt_batch.to(device)
        preds_logits = model(Xt_batch).detach().cpu().numpy() # Model outputs logits
        y_pred_logits.append(preds_logits)
        y_true_batches.append(yt_batch.numpy()) # Keep original true labels (one-hot or dummy)

        if (i + 1) % 10 == 0:
            print(f"Processed {((i + 1) * loader.batch_size):,} samples")

y_pred_logits = np.vstack(y_pred_logits)
y_true_combined = np.vstack(y_true_batches) # Combine true labels (will contain dummy if no labels were found)

# Save predictions (these are raw logits)
print("Saving predictions (raw logits)...")
output_filename_base = Path(args.data_path).stem.replace('X', '') # e.g., 'test_for_dssp'
prediction_output_path = f"y_pred_logits_{output_filename_base}.csv" # Specific name for output
with open(prediction_output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(y_pred_logits)
print(f"{prediction_output_path} saved")

#  Compute metrics if actual labels were loaded 
if y_true_data_for_metrics is not None: # Check if true labels were actually available for the dataset
    print("\n" + "-"*30)
    print("Calculating Classification Metrics")
    print("-" * 30)

    # For predictions, use argmax on logits to get the most likely class.
    y_pred_indices = np.argmax(y_pred_logits, axis=1)
    # For true labels, use argmax on the one-hot encoded true labels to get the class index.
    y_true_indices = np.argmax(y_true_combined, axis=1)

    #  Confusion Matrix 
    print("\nCalculating confusion matrix...")
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=np.arange(n_outputs))
    cm_normalized = cm.astype(np.float32)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Avoid division by zero for classes with no true samples
    cm_normalized = cm_normalized / row_sums * 100

    print("Confusion matrix (normalized %, sklearn):")
    print(cm_normalized)
    cm_output_path = f"confusion_matrix_{output_filename_base}.csv"
    np.savetxt(cm_output_path, cm_normalized, fmt='%7.3f', delimiter='\t')
    print(f"{cm_output_path} saved")

    #  Heatmap 
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap='Blues',
                xticklabels=np.arange(n_outputs), yticklabels=np.arange(n_outputs))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix (%)")
    plt.tight_layout()
    cm_plot_path = f"confusion_matrix_plot_{output_filename_base}.png"
    plt.savefig(cm_plot_path)
    plt.show()

    #  Distribution of True Classes 
    class_counts = collections.Counter(y_true_indices)
    print("\nClass Distribution (True Labels):")
    for label in sorted(class_counts):
        print(f"Class {label:2d}: {class_counts[label]} samples")

    #  Classification Report 
    print("\nClassification Report:")
    report = classification_report(
        y_true_indices,
        y_pred_indices,
        labels=np.arange(n_outputs),
        digits=3,
        output_dict=True,
        zero_division='warn' # Changed to 'warn' to avoid errors for missing classes
    )

    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")

    report_output_path = f"classification_report_{output_filename_base}.json"
    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"{report_output_path} saved.")

else:
    print("\nNo true labels were found, so confusion matrix and classification report cannot be generated.")