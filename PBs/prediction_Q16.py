import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from CNN_Q16 import activations, create_dataloader
from pathlib import Path # Import Path for file system operations
import re
import sys # Added for sys.exit

db_folder_name='learn_Q16'

# --- Function to find and load model parameters from filename ---
def find_and_load_q16_index_params(model_file_path, model_prefix="Q16"):
    """
    Extracts model architecture parameters from a Q16 model filename.
    It expects a direct file path and parses its name.
    """
    file_name = Path(model_file_path).name # Get just the filename

    # Regex to capture all parts of the Q16 model filename
    # Example: Q16_LAST_MODEL_3_5_3_2_1_256_tanh_leakyrelu_75.80_FOR_CPU
    q16_model_regex = re.compile(
        rf"^{re.escape(model_prefix)}_([A-Z_]+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_([a-zA-Z]+)_([a-zA-Z]+)_(\d+\.\d+)(?:_FOR_CPU)$"
    )

    match = q16_model_regex.match(file_name)
    if match:
        # Extract values (note: group 1 is model_type like LAST_MODEL/BEST_MODEL, we start from group 2 for params)
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
# --- End of function ---


# Model definition
class Model(nn.Module):
    def __init__(self, conv_layers, fc_layers, kernel_size, stride, padding,
                 first_layer_channels, conv_activation_name,
                 fc_activation_name, n_outputs=16):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        # Get activation functions from the global dictionary based on names
        self.conv_activation = activations[conv_activation_name]
        self.fc_activation = activations[fc_activation_name]
        length = 380  # input vector length
        ch = first_layer_channels

        for i in range(conv_layers):
            in_channels = 1 if i == 0 else ch
            conv = nn.Conv1d(in_channels, ch, kernel_size, stride, padding)
            bn = nn.BatchNorm1d(ch)
            self.convs.append(conv)
            self.bns.append(bn)
            # Calculate output length after convolution
            length = int((length + 2 * padding - kernel_size) / stride + 1)

        flat_dim = length * ch
        
        # Ensure flat_dim is valid before calculating powers for FC layers
        if flat_dim <= 1:
            raise ValueError(f"Calculated flat_dim after convolutional layers is {flat_dim}. "
                             "It must be greater than 1 to correctly determine FC layer dimensions. "
                             "Adjust conv_layers, kernel_size, stride, or padding.")

        # Dynamically calculate dimensions for fully connected layers
        # This mirrors the logic in the training script's MultiOutputRegression
        powers = np.linspace(
            int(np.floor(np.log2(flat_dim - 1))),
            int(np.ceil(np.log2(n_outputs + 1))),
            fc_layers - 1
        )
        dims = [flat_dim] + [int(2 ** p) for p in powers] + [n_outputs]

        for i in range(fc_layers):
            self.fcs.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.conv_activation(bn(conv(x)))
        x = x.flatten(start_dim=1)
        for fc in self.fcs[:-1]:
            x = self.fc_activation(fc(x))
        return self.fcs[-1](x)

# Accuracy function
def calculate_accuracy(y_true_onehot, y_pred_softmax):
    # y_pred_softmax contains probabilities from the Softmax output
    # For classification, we take argmax to get the predicted class index
    y_pred_labels = np.argmax(y_pred_softmax, axis=1)
    
    # y_true_onehot contains RMSD values, so the true label is the index of the minimum RMSD
    y_true_labels = np.argmin(y_true_onehot, axis=1) # Corrected to argmin for true labels

    return np.mean(y_pred_labels == y_true_labels)

# Main
def main():
    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument("--model_path", type=str,
                         default='./Q16_BEST_MODEL_7_5_3_2_1_512_tanh_leakyrelu_78.52_FOR_CPU',
                         help="Path to the trained model state_dict file.")
    parser.add_argument("--dataset_paths", nargs='+',
                         default=[f'./{db_folder_name}/train_Q16X.npy',
                                  f'./{db_folder_name}/test_Q16X.npy', # Use f-string for db_folder_name
                                  f'./{db_folder_name}/validation_Q16X.npy', # Use f-string
                                  f'./{db_folder_name}/CB513X.npy'], # CB513X is likely in alldb folder as well
                         help="List of dataset paths (e.g., './train_Q16X.npy ./val_Q16X.npy')")
    # Removed --conv_activation and --fc_activation args, as they will be parsed from filename
    # If you still want to allow overriding from cmd line, you can keep them and add logic.

    args = parser.parse_args()

    device = torch.device("cpu") # For inference, CPU is often sufficient

    # --- Updated: Extract model architecture from filename using the helper function ---
    (conv_layers, fc_layers, kernel_size, stride, padding,
     first_layer_channels, conv_activation_name, fc_activation_name, accuracy) = \
        find_and_load_q16_index_params(args.model_path)

    if conv_layers is None:
        print(f"Error: Could not parse architecture parameters from model filename: {args.model_path}")
        print("Ensure filename matches expected pattern (e.g., Q16_BEST_MODEL_3_5_3_2_1_256_tanh_leakyrelu_75.80_FOR_CPU).")
        # Fallback to some hardcoded defaults or exit
        # For a robust script, it's better to exit if the model definition cannot be inferred.
        sys.exit(1)

    # Load model using the extracted parameters
    model = Model(
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        first_layer_channels=first_layer_channels,
        conv_activation_name=conv_activation_name, # Use parsed activation name
        fc_activation_name=fc_activation_name      # Use parsed activation name
    ).to(device)

    # Load state_dict
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    # Use map_location=device to ensure it loads correctly regardless of original device
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    print(f"Loaded model from: {args.model_path}")
    print(f"Model Architecture: Conv Layers={conv_layers}, FC Layers={fc_layers}, Kernel={kernel_size}, Stride={stride}, Padding={padding}, Channels={first_layer_channels}")
    print(f"Using Conv Activation: '{conv_activation_name}', FC Activation: '{fc_activation_name}'\n")
    print(f"Reported Accuracy in filename: {accuracy:.2f}%\n")


    for data_path in args.dataset_paths:
        print(f"Processing dataset: {data_path}")
        if not os.path.exists(data_path):
            print(f"Data file missing: {data_path}, skipping...\n")
            continue

        X = np.load(data_path).astype('float32')
        # Ensure X has correct shape (batch_size, channels, length) for Conv1d
        if X.ndim == 2: # if it's (samples, features)
            X = X.reshape(X.shape[0], 1, X.shape[1]) # Reshape to (batch_size, 1, sequence_length)
        elif X.ndim == 3 and X.shape[1] != 1:
            print(f"Warning: Input data {data_path} has shape {X.shape}. Model expects 1 input channel.")
            X = X[:, 0:1, :] # Take only the first channel, keep dimension

        Y_path = data_path.replace('X.npy', 'Y.npy') # Default replacement
        # Specific overrides for your setup
        if "CB513X.npy" in data_path:
            Y_path = f"./{db_folder_name}/CB513Y.npy" # Assuming CB513Y is in alldb folder as well
        elif "train_Q16X.npy" in data_path: 
            Y_path = "./train_Q16Y.npy" # Assuming train_Q16Y.npy is in root

        if not os.path.exists(Y_path):
            print(f"Label file missing for {data_path}: {Y_path}, skipping...\n")
            continue

        y = np.load(Y_path).astype('float32')

        # DataLoader batch size can be large for inference as no gradients are calculated
        loader = create_dataloader(X, y, 10000)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (Xt_batch, yt_batch) in enumerate(loader):
                Xt_batch = Xt_batch.to(device)
                preds = model(Xt_batch).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(yt_batch.numpy())
                if (i + 1) % 10 == 0:
                    print(f"Processed {((i + 1) * loader.batch_size):,} samples")

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

        accuracy = calculate_accuracy(y_true, y_pred)
        print(f"Accuracy for {data_path}: {accuracy * 100:.2f}%\n")


if __name__ == "__main__":
    main()