import torch
import torch.nn as nn
import argparse
import numpy as np
import csv
from CNN_Q16 import activations, create_dataloader # Assuming activations is still exported
from sklearn.metrics import classification_report
import collections
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
from pathlib import Path # Import Path for file operations

#  Function to find and load model parameters from filename (copied from CNN_Q16.py) 
def find_and_load_q16_index_params(model_file_path, model_prefix="Q16"):
    """
    Extracts model architecture parameters from a Q16 model filename.
    It expects a direct file path and parses its name.

    Args:
        model_file_path (str or Path): The full path to the model file.
        model_prefix (str): The prefix of the model filenames to search for. Defaults to 'Q16'.

    Returns:
        tuple: A tuple containing (conv_layers, fc_layers, kernel_size, stride,
               padding, first_layer_channels, conv_activation, fc_activation, accuracy).
               Returns (None, None, ...) if the filename does not match the expected pattern.
    """
    file_name = Path(model_file_path).name # Get just the filename

    # Regex to capture all parts of the Q16 model filename
    # Example: Q16_BEST_MODEL_3_5_3_2_1_256_tanh_leakyrelu_75.80_FOR_CPU
    # This regex is robust to both _FOR_CPU and non-_FOR_CPU versions
    q16_model_regex = re.compile(
        rf"^{re.escape(model_prefix)}_([A-Z_]+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_([a-zA-Z]+)_([a-zA-Z]+)_(\d+\.\d+)(?:_FOR_CPU)?$"
    )

    match = q16_model_regex.match(file_name)
    if match:
        # Extract values
        conv_layers = int(match.group(2))
        fc_layers = int(match.group(3))
        kernel_size = int(match.group(4))
        stride = int(match.group(5))
        padding = int(match.group(6))
        first_layer_channels = int(match.group(7))
        conv_activation = match.group(8)
        fc_activation = match.group(9)
        accuracy = float(match.group(10))

        return (conv_layers, fc_layers, kernel_size, stride, padding,
                first_layer_channels, conv_activation, fc_activation, accuracy)
    else:
        return (None, None, None, None, None, None, None, None, None) # Return Nones if not found
#  End of function 


parser = argparse.ArgumentParser(description="Model Inference Script")
parser.add_argument("--model_path", type=str,
                    default='./Q16_BEST_MODEL_7_5_3_2_1_512_tanh_leakyrelu_78.52_FOR_CPU', # Updated default for new format
                    help="Path to the trained model state_dict file.")
parser.add_argument("--data_path", type=str,
                    default='./alldb/CB513X.npy',
                    help="Path to the input data (X.npy) for inference.")
# Removed --conv_activation and --fc_activation args, as they will be parsed from filename
args = parser.parse_args()

device = torch.device("cpu")

#  Model Architecture Parsing (UPDATED to use find_and_load_q16_index_params) 
(conv_layers, fc_layers, kernel_size, stride, padding,
 first_layer_channels, conv_activation_name, fc_activation_name, accuracy_from_filename) = \
    find_and_load_q16_index_params(args.model_path)

if conv_layers is None:
    print(f"Error: Could not parse architecture parameters from model filename: {args.model_path}")
    print("Ensure filename matches expected pattern (e.g., Q16_BEST_MODEL_3_5_3_2_1_256_tanh_leakyrelu_75.80_FOR_CPU).")
    # Fallback to some hardcoded defaults or exit
    # For a robust script, it's better to exit if the model definition cannot be inferred.
    sys.exit(1) # Exit the script as model architecture is unknown

n_outputs = 16 # Q16 problem has 16 output classes

# Model definition (now uses parsed parameters for initialization)
class Model(nn.Module):
    def __init__(self, conv_layers, fc_layers, kernel_size, stride, padding,
                 first_layer_channels, conv_activation_name, fc_activation_name,
                 n_outputs=16, input_len=380, freeze_conv=False):
        super().__init__()

        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.first_layer_channels = first_layer_channels
        self.n_outputs = n_outputs
        self.input_len = input_len

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.conv_activation = activations[conv_activation_name]
        self.fc_activation = activations[fc_activation_name]

        l = self.input_len
        for i in range(self.conv_layers):
            ch = self.first_layer_channels
            in_channels = 1 if i == 0 else ch
            conv = nn.Conv1d(in_channels, ch, self.kernel_size, self.stride, self.padding)
            bn = nn.BatchNorm1d(ch)
            self.convs.append(conv)
            self.bns.append(bn)
            l = int((l + 2 * self.padding - self.kernel_size) / self.stride + 1)

        flat_dim = l * self.first_layer_channels # This should use self.first_layer_channels
        if flat_dim <= 1:
            raise ValueError(f"Invalid flattened dimension after convolution layers: {flat_dim}")

        powers = np.linspace(
            int(np.floor(np.log2(flat_dim - 1))),
            int(np.ceil(np.log2(self.n_outputs + 1))),
            self.fc_layers - 1
        )
        values = [flat_dim] + [int(2 ** p) for p in powers] + [self.n_outputs]

        for i in range(self.fc_layers):
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
        for i in range(self.conv_layers): # Use self.conv_layers
            x = self.conv_activation(self.bns[i](self.convs[i](x)))
        x = x.flatten(start_dim=1)
        for i in range(len(self.fcs) - 1):
            x = self.fc_activation(self.fcs[i](x))
        return self.fcs[-1](x) 

def load_model_from_params(model_path, conv_layers, fc_layers, kernel_size, stride,
                           padding, first_layer_channels, conv_activation_name, fc_activation_name, n_outputs):
    model = Model(
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        first_layer_channels=first_layer_channels,
        conv_activation_name=conv_activation_name,
        fc_activation_name=fc_activation_name,
        n_outputs=n_outputs
    )
    # Check if the model file exists before loading
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1) # Exit if the model file doesn't exist

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Load model using parsed parameters
model = load_model_from_params(
    args.model_path, conv_layers, fc_layers, kernel_size, stride,
    padding, first_layer_channels, conv_activation_name, fc_activation_name, n_outputs
).to(device)

print(f"Loaded model from: {args.model_path}")
print(f"Model Architecture: Conv Layers={conv_layers}, FC Layers={fc_layers}, Kernel={kernel_size}, Stride={stride}, Padding={padding}, Channels={first_layer_channels}")
print(f"Using Conv Activation: '{conv_activation_name}', FC Activation: '{fc_activation_name}'\n")

X = np.load(args.data_path).astype('float32')
X = X.reshape(X.shape[0], 1, X.shape[-1])

Y_path = args.data_path.replace('X.npy', 'Y.npy') # More robust way to derive Y_path
# Special handling for CB513 if it's expected to be in a specific folder, e.g., 'alldb'
if "CB513X.npy" in args.data_path:
    # Assuming CB513Y.npy is in the same directory as CB513X.npy
    Y_path = Path(args.data_path).parent / "CB513Y.npy"


has_labels = os.path.exists(Y_path)
# Removed the `has_rmsd` check for `yt.csv` as it seems to be an old logic.
# The primary source for true labels for Q16 should be the corresponding Y.npy file.

y_true_rmsd_data = None # Store the actual RMSD array if available

if has_labels:
    y_true_rmsd_data = np.load(Y_path).astype('float32')
    # For dataloader, we pass the RMSD array.
    loader = create_dataloader(X, y_true_rmsd_data, 1024)
else:
    y_true_rmsd_data = None
    print(f"No labels found for {args.data_path} at {Y_path} — only generating predictions.")
    # If no labels, create a DataLoader with dummy Y for consistency in loop structure
    from torch.utils.data import TensorDataset, DataLoader
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
output_dir = "inference_results"
os.makedirs(output_dir, exist_ok=True)
logits_filename = Path(args.model_path).stem.replace('_FOR_CPU', '') + "_" + Path(args.data_path).stem.replace('X', '') + "_logits.csv"
logits_path = Path(output_dir) / logits_filename

with open(logits_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(y_pred_logits)
print(f"Raw logits saved to {logits_path}")

#  Compute metrics if labels are available 
if y_true_rmsd_data_combined is not None: # Check if true labels were available for the dataset
    print("\n" + "-"*30)
    print("Calculating Classification Metrics")
    print("-" * 30)

    # For predictions, use argmax on logits to get the most likely class.
    y_pred_indices = np.argmax(y_pred_logits, axis=1)
    # For true labels, use argmin on RMSD values to get the true class.
    y_true_indices = np.argmin(y_true_rmsd_data_combined, axis=1)

    #  Confusion Matrix 
    print("\nCalculating confusion matrix...")
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=np.arange(n_outputs))
    cm_normalized = cm.astype(np.float32)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Avoid division by zero for classes with no true samples
    cm_normalized = cm_normalized / row_sums * 100

    print("Confusion matrix (normalized %, sklearn):")
    print(cm_normalized)
    
    cm_filename = Path(args.model_path).stem.replace('_FOR_CPU', '') + "_" + Path(args.data_path).stem.replace('X', '') + "_confusion_matrix.csv"
    cm_path = Path(output_dir) / cm_filename
    np.savetxt(cm_path, cm_normalized, fmt='%7.3f', delimiter='\t')
    print(f"Confusion matrix saved to {cm_path}")

    #  Heatmap 
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap='Blues',
                xticklabels=np.arange(n_outputs), yticklabels=np.arange(n_outputs))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Normalized Confusion Matrix (%) for {Path(args.data_path).stem.replace('X','')}")
    plt.tight_layout()
    cm_plot_filename = Path(args.model_path).stem.replace('_FOR_CPU', '') + "_" + Path(args.data_path).stem.replace('X', '') + "_confusion_matrix_plot.png"
    cm_plot_path = Path(output_dir) / cm_plot_filename
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")
    # plt.show() # Uncomment if you want the plot to display immediately

    #  Distribution of True Classes 
    class_counts_true = collections.Counter(y_true_indices) # Renamed to avoid conflict
    print("\nРаспределение классов (реальные данные):")
    for label in sorted(class_counts_true):
        print(f"Класс {label:2d}: {class_counts_true[label]} примеров")

    #  Classification Report 
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

    report_filename = Path(args.model_path).stem.replace('_FOR_CPU', '') + "_" + Path(args.data_path).stem.replace('X', '') + "_classification_report.json"
    report_path = Path(output_dir) / report_filename
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to {report_path}.")

else:
    print("\nNo true labels were found, so confusion matrix and classification report cannot be generated.")