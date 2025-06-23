import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from CNN_Q8 import activations, create_dataloader 
import re
from pathlib import Path

# Global constant for the database folder name
db_folder_name = 'learn_Q8'

# Model definition
class Model(nn.Module):
    def __init__(self, conv_layers, fc_layers, kernel_size, stride, padding, 
                 first_layer_channels, conv_activation_name='tanh', 
                 fc_activation_name='leakyrelu', n_outputs=8, input_len=380): 
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.conv_activation = activations[conv_activation_name]
        self.fc_activation = activations[fc_activation_name]

        length = input_len  # input vector length
        ch = first_layer_channels

        for i in range(conv_layers):
            in_channels = 1 if i == 0 else ch
            conv = nn.Conv1d(in_channels, ch, kernel_size, stride, padding)
            bn = nn.BatchNorm1d(ch)
            self.convs.append(conv)
            self.bns.append(bn)
            length = int((length + 2 * padding - kernel_size) / stride + 1)

        flat_dim = length * ch
        
        if flat_dim <= 1:
            raise ValueError(f"Calculated flat_dim after convolutional layers is {flat_dim}. "
                             "It must be greater than 1 to correctly determine FC layer dimensions. "
                             "Adjust conv_layers, kernel_size, stride, or padding.")

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
        return self.fcs[-1](x) # Return raw logits for classification

# Accuracy function
def calculate_accuracy(y_true_onehot, y_pred_raw_logits): 
    y_pred_labels = np.argmax(y_pred_raw_logits, axis=1)
    y_true_labels = np.argmax(y_true_onehot, axis=1) 
    return np.mean(y_pred_labels == y_true_labels)

# Function to extract model parameters from filename
def extract_model_params_from_filename(model_path):
    match = re.search(r"_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_([a-zA-Z]+)_([a-zA-Z]+)_(\d+\.\d+)", Path(model_path).name)
    if match:
        conv_layers = int(match.group(1))
        fc_layers = int(match.group(2))
        kernel_size = int(match.group(3))
        stride = int(match.group(4))
        padding = int(match.group(5))
        first_layer_channels = int(match.group(6))
        conv_activation_name = match.group(7) # Added for more robust parsing
        fc_activation_name = match.group(8)   # Added for more robust parsing
        return conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels, conv_activation_name, fc_activation_name
    return None, None, None, None, None, None, None, None

# Main 
def main():
    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument("--model_path", type=str, 
                        default='./Q8_BEST_MODEL_7_5_3_2_1_512_tanh_leakyrelu_68.48_FOR_CPU', # Corrected default path
                        help="Path to the trained model state_dict file.")
    
    parser.add_argument("--dataset_paths", nargs='+', 
                        default=['./train_Q8X.npy', 
                                 f'./{db_folder_name}/test_for_dsspX.npy', 
                                 f'./{db_folder_name}/validation_for_dsspX.npy', 
                                 f'./{db_folder_name}/CB513X.npy'], 
                        help="List of dataset paths (e.g., './train_Q8X.npy ./alldb/test_for_dsspX.npy')")
    
    args = parser.parse_args()

    device = torch.device("cpu") # For inference, CPU is often sufficient

    # Initialize with default values, which will be overridden if parsing is successful
    conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels = 2, 1, 10, 2, 1, 32
    conv_activation = 'tanh' # Default value if not parsed
    fc_activation = 'leakyrelu' # Default value if not parsed

    # Attempt to extract parameters from the filename
    parsed_params = extract_model_params_from_filename(args.model_path)
    if all(p is not None for p in parsed_params):
        conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels, conv_activation, fc_activation = parsed_params
        print("Parsed model parameters from filename successfully.")
    else:
        print("Warning: Could not parse all architecture parameters or activation functions from model filename. "
              "Using hardcoded defaults. Ensure filename matches expected pattern (e.g., Q8_BEST_MODEL_..._2_1_10_2_1_32_tanh_leakyrelu_...).")

    # Determine input length from the first dataset
    try:
        sample_X = np.load(args.dataset_paths[0]).astype('float32')
        input_length = sample_X.shape[-1] # Assuming last dim is sequence length
        print(f"Detected input sequence length from first dataset: {input_length}")
    except Exception as e:
        print(f"Error determining input length from first dataset: {e}. Defaulting to 380.")
        input_length = 380 # Fallback default

    model = Model(
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        first_layer_channels=first_layer_channels,
        conv_activation_name=conv_activation, 
        fc_activation_name=fc_activation,
        n_outputs=8, # Q8 implies 8 output classes
        input_len=input_length
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() 

    print(f"Loaded model from: {args.model_path}")
    print(f"Model Architecture: Conv Layers={conv_layers}, FC Layers={fc_layers}, Kernel={kernel_size}, Stride={stride}, Padding={padding}, Channels={first_layer_channels}")
    print(f"Using Conv Activation: '{conv_activation}', FC Activation: '{fc_activation}'\n")

    for data_path in args.dataset_paths:
        print(f"Processing dataset: {data_path}")
        X_data = np.load(data_path).astype('float32')
        
        if X_data.ndim == 2: # if it's (samples, features)
            X_data = X_data.reshape(X_data.shape[0], 1, X_data.shape[-1])
        elif X_data.ndim == 3 and X_data.shape[1] != 1: 
            # if it's (samples, N_channels, features) where N_channels > 1 but model expects 1
            print(f"Warning: Input data {data_path} has shape {X_data.shape}. Model expects 1 input channel. Using first channel.")
            X_data = X_data[:, 0:1, :] # Take only the first channel, keep dimension
        base_name = Path(data_path).stem.replace('X', 'Y') # e.g., 'train_Q8X' -> 'train_Q8Y'
        Y_path = Path(data_path).parent / f"{base_name}.npy"
        
        if not Y_path.exists():
            # Special handling for CB513 which might be directly in the root or a specific folder
            if "CB513X.npy" in data_path:
                Y_path = Path(data_path).parent / "CB513Y.npy" # Assuming CB513Y.npy is in the same folder
                if not Y_path.exists():
                    # If CB513 is in alldb but CB513Y is not, try the root
                    Y_path = Path("./CB513Y.npy")
            
            if not Y_path.exists():
                print(f"Label file missing for {data_path}: {Y_path}, skipping...\n")
                continue

        y_data = np.load(Y_path).astype('float32')
        
        loader = create_dataloader(X_data, y_data, 10000) # Use a large batch size for inference

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (Xt_batch, yt_batch) in enumerate(loader):
                Xt_batch = Xt_batch.to(device)
                preds = model(Xt_batch).cpu().numpy() # Model outputs raw logits
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