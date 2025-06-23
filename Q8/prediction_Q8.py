import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from CNN_Q8 import activations, create_dataloader
import re
from pathlib import Path

# Global constant for the database folder name
db_folder_name = 'alldb'

# Model definition
class Model(nn.Module):
    def __init__(self, conv_layers, fc_layers, kernel_size, stride, padding, 
                 first_layer_channels, conv_activation_name='tanh', 
                 fc_activation_name='leakyrelu', n_outputs=8): # Changed n_outputs to 8 for Q8 model
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        # Get activation functions from the global dictionary based on names
        self.conv_activation = activations[conv_activation_name]
        self.fc_activation = activations[fc_activation_name]

        # Note: 'length' (input vector length) and 'y.shape[-1]' (n_outputs) are used
        # in the training script's Model definition to calculate layer dimensions.
        # Here, for inference, assuming a fixed input length of 380 and n_outputs=8
        # for this specific Q8 model. If these vary, they should also be passed as args
        # or derived dynamically.
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
def calculate_accuracy(y_true_onehot, y_pred_raw_logits): # Renamed y_pred_rmsd to y_pred_raw_logits for clarity
    # y_pred_raw_logits contains raw model outputs (logits)
    # For classification, we take argmax to get the predicted class index
    y_pred_labels = np.argmax(y_pred_raw_logits, axis=1)
    
    # y_true_onehot is assumed to contain one-hot encoded labels,
    # so the true label is the index of the maximum value (1)
    y_true_labels = np.argmax(y_true_onehot, axis=1) # Changed from argmin to argmax for true labels

    return np.mean(y_pred_labels == y_true_labels)

# Main 
def main():
    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument("--model_path", type=str, 
                        default='./Q8_BEST_MODEL_7_5_3_2_1_512_68.48_FOR_CPU', # Updated default model path for Q8
                        help="Path to the trained model state_dict file.")
    parser.add_argument("--dataset_paths", nargs='+', 
                        default=['./train_Q8X.npy', 
                                 f'./{db_folder_name}/test_for_dsspX.npy', # Updated dataset paths for Q8
                                 f'./{db_folder_name}/validation_for_dsspX.npy', # Updated dataset paths for Q8
                                 './CB513X.npy'], # CB513X.npy path might vary, keeping as is based on provided code
                        help="List of dataset paths (e.g., './train_Q8X.npy ./alldb/test_for_dsspX.npy')")
    parser.add_argument('--conv_activation', type=str, default='tanh', 
                        help="Activation function for convolutional layers (e.g., 'relu', 'tanh', 'sigmoid')")
    parser.add_argument('--fc_activation', type=str, default='leakyrelu', 
                        help="Activation function for fully connected layers (e.g., 'relu', 'tanh', 'sigmoid', 'leakyrelu')")
    
    args = parser.parse_args()

    device = torch.device("cpu") # For inference, CPU is often sufficient

    try:
        parts = Path(args.model_path).stem.split('_') # .stem gets filename without extension
        # e.g., 'Q8_BEST_MODEL_2_1_10_2_1_32_68.48_FOR_CPU'
        match_arch_params = re.search(r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", Path(args.model_path).name)
        if match_arch_params:
            conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels = map(int, match_arch_params.groups())
        else:
            print("Warning: Could not parse architecture parameters from model filename. Using hardcoded defaults.")
            # Fallback to some defaults or raise an error if parsing fails
            conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels = 2, 1, 10, 2, 1, 32

    except ValueError as e:
        print(f"Error parsing model architecture from filename: {e}. Ensure filename matches expected pattern (e.g., Q8_BEST_MODEL_2_1_10_2_1_32_...). Using default parameters.")
        # Fallback to some reasonable defaults if parsing completely fails
        conv_layers, fc_layers, kernel_size, stride, padding, first_layer_channels = 2, 1, 10, 2, 1, 32


    # Load model using the extracted/default parameters and the parsed activation function names
    model = Model(
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        first_layer_channels=first_layer_channels,
        conv_activation_name=args.conv_activation, # Pass argument-specified activation name
        fc_activation_name=args.fc_activation    # Pass argument-specified activation name
    ).to(device)
    
    # Load state_dict only
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    print(f"Loaded model from: {args.model_path}")
    print(f"Model Architecture: Conv Layers={conv_layers}, FC Layers={fc_layers}, Kernel={kernel_size}, Stride={stride}, Padding={padding}, Channels={first_layer_channels}")
    print(f"Using Conv Activation: '{args.conv_activation}', FC Activation: '{args.fc_activation}'\n")

    for data_path in args.dataset_paths:
        print(f"Processing dataset: {data_path}")
        X = np.load(data_path).astype('float32')
        # Ensure X has correct shape (batch_size, channels, length)
        # Original script reshapes to (X.shape[0], 1, X.shape[-1])
        # Assuming input to Model is (batch_size, 1, sequence_length)
        if X.ndim == 2: # if it's (samples, features)
            X = X.reshape(X.shape[0], 1, X.shape[-1])
        elif X.ndim == 3 and X.shape[1] != 1: # if it's (samples, features, 1) and needs reshaping to (samples, 1, features)
            # This case might need careful handling depending on original data shape
            # For this model, input channel is 1, so ensure the second dimension is 1.
            print(f"Warning: Input data {data_path} has shape {X.shape}. Expected (batch, 1, features). Attempting to reshape if needed.")
            if X.shape[-1] == 1: # if it's (batch, features, 1), make it (batch, 1, features)
                X = X.transpose(0, 2, 1)
            elif X.shape[1] > 1: # if it's (batch, N_channels, features) where N_channels > 1 but model expects 1
                print("Warning: Model expects 1 input channel, but data has multiple channels. Using first channel.")
                X = X[:, 0:1, :] # Take only the first channel, keep dimension
        
        # Construct Y_path based on X_path
        Y_path_base = data_path.replace('X.npy', 'Y.npy')
        Y_path = Y_path_base
        
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
