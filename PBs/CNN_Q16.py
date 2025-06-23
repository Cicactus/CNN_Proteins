import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from collections import deque
import numpy as np
import os
import statistics
import argparse
import csv
import re
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Global Constants
LastModelName = "trained_models/Q16_LAST_MODEL"
BestModelName = "Q16_BEST_MODEL"

# Argument Parser
def get_arguments():
    parser = argparse.ArgumentParser(description="argparse")

    parser.add_argument("--lr_start", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument('--zero_grad_batch', type=int, default=2048, help="Effective batch size, determining gradient update frequency (must be >= batch size)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--conv_layers", type=int, default=2, help="Number of convolutional layers")
    parser.add_argument("--fc_layers", type=int, default=1, help="Number of fully connected layers")
    parser.add_argument("--kernel_size", type=int, default=10, help="Kernel size")
    parser.add_argument('--stride', type=int, default=2, help="Stride value")
    parser.add_argument('--padding', type=int, default=1, help="Padding value")
    parser.add_argument('--conv_activation', type=str, default='tanh', help="Activation function for convolutional layers")
    parser.add_argument('--fc_activation', type=str, default='sigmoid', help="Activation function for fully connected layers")
    parser.add_argument('--first_layer_channels', type=int, default=32, help="Number of output channels for the first convolutional layer")
    parser.add_argument('--accuracy_datasets', type=str, default='test CB513', help="Datasets on which accuracy is measured (space-separated)")
    parser.add_argument('--e_acc', type=int, default=10, help="Number of epochs after which accuracy is re-measured")
    parser.add_argument('--pretrained', type=str, default='n', help="Load a pretrained model (Y/n)")
    parser.add_argument('--l1_lambda', type=float, default=0.00001, help="L1 regularization lambda")
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer to use")
    parser.add_argument('--loss', type=str, default='mse_loss', help="Функция потерь")


    args = parser.parse_args()
    return args

# Dictionaries for Activations and Optimizers
activations = {
    "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "softmax": nn.Softmax(dim=1),
    "linear": nn.Identity(), "softplus": nn.Softplus(), "leakyrelu": nn.LeakyReLU(), "prelu": nn.PReLU(),
    "rrelu": nn.RReLU(), "selu": nn.SELU(), "celu": nn.CELU(), "glu": nn.GLU(), "elu": nn.ELU(),
    "hardshrink": nn.Hardshrink(), "hardsigmoid": nn.Hardsigmoid(), "hardtanh": nn.Hardtanh(),
    "hardswish": nn.Hardswish(), "logsigmoid": nn.LogSigmoid(), "relu6": nn.ReLU6(),
    "silu": nn.SiLU(), "mish": nn.Mish(), "softshrink": nn.Softshrink(), "softsign": nn.Softsign(),
    "tanhshrink": nn.Tanhshrink(), "threshold": nn.Threshold(0, 0), "softmin": nn.Softmin(dim=1),
    "softmax2d": nn.Softmax2d(), "logsoftmax": nn.LogSoftmax(dim=1),
}

optimizers = {
    "adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad, "adamw": torch.optim.AdamW, "nadam": torch.optim.NAdam,
    "adamax": torch.optim.Adamax, "asgd": torch.optim.ASGD,
}

#  DataLoader Creation
def create_dataloader(X_data, y_data, batch_size):
    dataset = TensorDataset(torch.from_numpy(X_data), torch.from_numpy(y_data))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model Loading and Hyperparameter Check Helper
def load_model_with_partial_weights(new_model, state_dict_path):
    old_state = torch.load(state_dict_path)
    new_state = new_model.state_dict()
    matched_state = {k: v for k, v in old_state.items() if k in new_state and v.size() == new_state[k].size()}
    new_state.update(matched_state)
    new_model.load_state_dict(new_state)
    skipped_layers = {k: v.size() for k, v in old_state.items() if k not in matched_state}
    return skipped_layers

def check_hyperparameter_match(old_state, new_state):
    # This checks if the shapes of parameters in old_state match new_state
    # If they don't match for any common layer, it suggests hyperparameters changed.
    for k in new_state:
        if k in old_state and old_state[k].size() != new_state[k].size():
            return False
    return True

def get_loader_accuracy(loader, model, device):
    """
    Calculates the average classification accuracy for a CNN model over a DataLoader.
    Expects yt_batch to contain RMSD values, from which class indices are derived using argmin.
    """
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        model.eval() # Ensure model is in eval mode
        for Xt_batch, yt_batch_rmsd in loader: # Renamed for clarity, yt_batch_rmsd contains RMSD values
            Xt_batch = Xt_batch.to(device)

            # Derive true class indices from RMSD values by finding the minimum
            true_labels_class_indices = torch.argmin(yt_batch_rmsd, dim=1).long().to(device)

            y_pred_logits = model(Xt_batch) # Model outputs raw logits

            # Compare predicted class (argmax of logits) with true class indices
            total_correct += (torch.argmax(y_pred_logits, dim=1) == true_labels_class_indices).sum().item()
            total_samples += true_labels_class_indices.size(0)
    
    if total_samples == 0:
        return 0.0
    return total_correct / total_samples

def get_all_accuracy_results(model, device, loaders, accuracy_datasets, is_gnn=False):
    """
    Computes accuracy for specified datasets.
    is_gnn is set to False by default as this script is for CNNs.
    """
    with torch.no_grad():
        model.eval()
        dataset_names = accuracy_datasets.split()
        accuracies = {}
        for dataset_name in dataset_names:
            if dataset_name in loaders:
                loader = loaders[dataset_name]
                # Always use get_loader_accuracy_CNN as this script is for CNNs
                acc = get_loader_accuracy(loader, model, device)
                accuracies[dataset_name] = acc
                print(f"{dataset_name} Accuracy: {acc * 100:.2f}%")
            else:
                print(f"Loader for dataset {dataset_name} not found!")
        return accuracies

#  File Operations Helpers
def delete_file(file_path):
    """Deletes a file if it exists."""
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def extract_cnn_model_info(folder_path):
    """
    Extracts model info (path, index, accuracy) for CNN models from a folder.
    """
    folder = Path(folder_path)
    model_info_list = []

    for file_path in folder.glob("*"):
        # Ensure it's a file, not a GNN model, and not a _FOR_CPU file
        if file_path.is_file() and "GNN" not in file_path.name and "_FOR_CPU" not in file_path.name:
            # Regex to match the full index (including activations) and accuracy at the end of the filename
            match = re.search(r"_(\d+_\d+_\d+_\d+_\d+_\d+_[a-zA-Z]+_[a-zA-Z]+)_(\d+\.\d+)$", file_path.name)
            if match:
                model_index = match.group(1)
                model_accuracy = float(match.group(2))
                model_info_list.append((file_path, model_index, model_accuracy))
    return model_info_list

#  Main Save and Accuracy Function
def save_data_and_calculate_accuracy(
    model, optimizer, epoch, epoch_loss, loaders, args, device, current_index, is_gnn=False
):
    """
    Calculates accuracies, logs results to CSV, and saves best/last models.
    The 'last model' for the current index is updated only if current accuracy is better.
    The 'overall best model' is saved only if it performs better.
    """
    # Define prefixes for saving models
    model_prefix = BestModelName # Overall best model
    last_model_prefix = LastModelName # Last model for the current index
    csv_file = "training_results.csv" # CSV log file
    extract_model_info_func = extract_cnn_model_info # Helper to find existing models

    with torch.no_grad():
        model.eval() # Set model to evaluation mode

        # Get accuracies for all specified datasets
        accuracies = get_all_accuracy_results(
            model, device, loaders, args.accuracy_datasets, is_gnn=False
        )

        test_accuracy = accuracies.get('test', 0)
        val_accuracy = accuracies.get('val', 0)
        CB513_accuracy = accuracies.get('CB513', 0)

        # Prepare data for CSV logging
        data = {
            "lr_start": args.lr_start, "batch_size": args.batch_size, "zero_grad_batch": args.zero_grad_batch,
            "epochs": epoch + 1, "conv_layers": args.conv_layers, "fc_layers": args.fc_layers,
            "conv_activation": args.conv_activation, "fc_activation": args.fc_activation,
            "loss_function": "CrossEntropyLoss",
            "first_layer_channels": args.first_layer_channels, "lr_final": optimizer.param_groups[0]['lr'],
            "final_loss": epoch_loss, "test_accuracy": test_accuracy,
            "validation_accuracy": val_accuracy, "CB513_accuracy": CB513_accuracy,
            "optimizer": args.optimizer, "l1_lambda": args.l1_lambda, "pretrained": args.pretrained,
            "accuracy_datasets": args.accuracy_datasets
        }
        data.update({"kernel_size": args.kernel_size, "stride": args.stride, "padding": args.padding})

        # Append results to CSV file
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader() # Write header if file is new
            writer.writerow(data) # Write the current epoch's data

        trained_models = extract_model_info_func("./trained_models")
        best_models = extract_model_info_func("./") # Assuming best overall models are saved in the root

        #  Logic for saving the 'Last Model' for the current index (overwrite only if better)
        existing_last_model_path = None
        existing_last_model_accuracy = -1.0 # Initialize to a value lower than any possible accuracy

        for file_path, index, accuracy_from_filename in trained_models:
            # Check if this file is a 'last' model for the current architecture index
            if index == current_index and file_path.name.startswith(Path(LastModelName).name):
                existing_last_model_path = file_path
                existing_last_model_accuracy = accuracy_from_filename
                break # Found the existing 'last' model for this index, no need to search further

        # Compare current validation accuracy with the existing 'last' model's accuracy
        if val_accuracy * 100 > existing_last_model_accuracy:
            if existing_last_model_path:
                # Delete the old 'last' model and its _FOR_CPU counterpart
                delete_file(existing_last_model_path)
                delete_file(Path(str(existing_last_model_path) + "_FOR_CPU"))
                print(f"Removed previous 'last' model for index {current_index} (Accuracy: {existing_last_model_accuracy:.2f}%).")

            # Save the current model as the new 'last' model
            torch.save(model, f'{last_model_prefix}_{current_index}_{val_accuracy * 100:.2f}')
            torch.save(model.state_dict(), f'{last_model_prefix}_{current_index}_{val_accuracy * 100:.2f}_FOR_CPU')
            print(f"Saved current model as the new 'last' model for index {current_index} (Validation Accuracy: {val_accuracy * 100:.2f}%).")
        else:
            print(f"Current model's accuracy ({val_accuracy * 100:.2f}%) is not better than the existing 'last' model for index {current_index} ({existing_last_model_accuracy:.2f}%). No update to 'last' model.")

        #  Logic for saving the 'Overall Best Model' (remains unchanged, always overwrite if better)
        overall_best_accuracy = -1.0 # Initialize with a value lower than any possible accuracy
        overall_best_model_path = None
        for file_path, index, accuracy_from_filename in best_models:
            if accuracy_from_filename > overall_best_accuracy:
                overall_best_accuracy = accuracy_from_filename
                overall_best_model_path = file_path
        
        if val_accuracy * 100 > overall_best_accuracy:
            if overall_best_model_path:
                delete_file(overall_best_model_path)
                delete_file(Path(str(overall_best_model_path) + "_FOR_CPU")) # Delete associated _FOR_CPU file
            
            torch.save(model, f'{model_prefix}_{current_index}_{val_accuracy * 100:.2f}')
            torch.save(model.state_dict(), f'{model_prefix}_{current_index}_{val_accuracy * 100:.2f}_FOR_CPU')
            print(f"Saved current model as the **overall best model** (Validation Accuracy: {val_accuracy * 100:.2f}%).")
        else:
            print(f"Current model's accuracy ({val_accuracy * 100:.2f}%) is not better than the overall best model ({overall_best_accuracy:.2f}%).")
    
    return True

# Get arguments
args = get_arguments()

# Hyperparameters output
print(f"Initial learning rate: {args.lr_start}")
print(f"Batch size: {args.batch_size}")
print(f"Batch size for gradient resets: {args.zero_grad_batch}")
print(f"Epochs: {args.epochs}")
print(f"Number of convolution layers: {args.conv_layers}")
print(f"Number of fully connected layers: {args.fc_layers}")
print(f"Kernel size: {args.kernel_size}")
print(f"Stride value: {args.stride}")
print(f"Padding value: {args.padding}")
print(f"Convolution activation function: {args.conv_activation}")
print(f"FC activation function: {args.fc_activation}")
print(f"Number of channels on a first conv layer: {args.first_layer_channels}")
print(f"L1 lambda: {args.l1_lambda}")
print(f"Optimizer: {args.optimizer}")

# UPDATED current_index to include activation functions
current_index = f'{args.conv_layers}_{args.fc_layers}_{args.kernel_size}_{args.stride}_{args.padding}_{args.first_layer_channels}_{args.conv_activation}_{args.fc_activation}'

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

db_folder_name='alldb'


X = np.load(f"train_Q16X.npy").astype('float32')
y = np.load(f"train_Q16Y.npy").astype('float32')
CB513X = np.load(f"./{db_folder_name}/CB513X.npy").astype('float32')
CB513Y = np.load(f"./{db_folder_name}/CB513Y.npy").astype('float32')
X_test = np.load(f"./{db_folder_name}/test_Q16X.npy").astype('float32')
y_test = np.load(f"./{db_folder_name}/test_Q16Y.npy").astype('float32')
X_val = np.load(f"./{db_folder_name}/validation_Q16X.npy").astype('float32')
y_val = np.load(f"./{db_folder_name}/validation_Q16Y.npy").astype('float32')

# Reshaping input data
X = X.reshape(X.shape[0], 1, X.shape[-1])
CB513X = CB513X.reshape(CB513X.shape[0], 1, CB513X.shape[-1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[-1])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[-1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Model Definition
class MultiOutputRegression(nn.Module):
    def __init__(self, freeze_conv=False):
        super().__init__()

        l = X.shape[-1]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.conv_activation = activations[args.conv_activation]
        self.fc_activation = activations[args.fc_activation]

        for i in range(args.conv_layers):
            channels_this_layer = args.first_layer_channels
            conv = nn.Conv1d(1 if i == 0 else channels_this_layer, channels_this_layer,
                             kernel_size=args.kernel_size, stride=args.stride, padding=args.padding)
            bn = nn.BatchNorm1d(channels_this_layer)
            self.convs.append(conv)
            self.bns.append(bn)
            l = int((l + 2 * args.padding - args.kernel_size) / args.stride + 1)

        l = int(l) * channels_this_layer
        if l <= 1:
            raise ValueError(f"Invalid input length after convolution layers: {l}. Adjust conv_layers or kernel_size.")

        # num_classes is determined by the last dimension of the y (RMSD) array
        num_classes = y.shape[-1]
        powers = np.linspace(int(np.floor(np.log2(l - 1))), int(np.ceil(np.log2(num_classes + 1))), args.fc_layers - 1)
        values = [l] + [int(2 ** p) for p in powers] + [num_classes]

        for i in range(args.fc_layers):
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
        for i in range(args.conv_layers):
            x = self.conv_activation(self.bns[i](self.convs[i](x)))
        x = x.flatten(start_dim=1)
        for i in range(len(self.fcs) - 1):
            x = self.fc_activation(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x


# Pretrained model handling
previous_model_yn = args.pretrained

if previous_model_yn == 'Y':
    print('Type in index of a model you want to take as a basis for training')
    model_index = input()

    state_dict_path = LastModelName + f'_{model_index}_FOR_CPU'

    if os.path.exists(state_dict_path):
        old_state = torch.load(state_dict_path)
        temp_model = MultiOutputRegression().to(device)
        temp_state = temp_model.state_dict()
        hyperparameters_match = check_hyperparameter_match(old_state, temp_state)

        model = MultiOutputRegression(freeze_conv=not hyperparameters_match).to(device)

        if hyperparameters_match:
            print("Hyperparameters match! All layers will remain unfrozen.")
        else:
            print("Hyperparameters do not match. Convolutional layers will be frozen.")

        skipped_layers = load_model_with_partial_weights(model, state_dict_path)
        print(f"Skipped layers due to mismatched sizes: {skipped_layers}")

        print('How many epochs to skip?')
        skip_epochs = int(input())
    else:
        raise ValueError(f"No state_dict file found at: {state_dict_path}")
elif previous_model_yn == 'n':
    model = MultiOutputRegression().to(device)
    skip_epochs = 0


num_classes = y.shape[-1] # The number of columns in y corresponds to the number of classes

# Get the class indices for the training data by taking argmin of RMSD values
y_train_indices_for_counts = np.argmin(y, axis=1).astype(int)

class_counts = Counter(y_train_indices_for_counts)

# Create weights tensor (inverse frequency)
# A small epsilon prevents true division by zero for actual zeros (though 0 counts are converted to 1e-6).
weights = torch.tensor([1.0 / (class_counts.get(i, 0) if class_counts.get(i, 0) > 0 else 1e-6) for i in range(num_classes)],
                         device=device, dtype=torch.float)

# Normalize weights so they sum to the number of classes (often helps stability)
weights = weights / weights.sum() * num_classes

print(f"Calculated class counts: {class_counts}")
print(f"Calculated class weights: {weights.tolist()}")

# Instantiate CrossEntropyLoss with weights
criterion = nn.CrossEntropyLoss(weight=weights)

# Optimizer and Scheduler
optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr_start)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=1,
    threshold=1e-4,
    min_lr=1e-7
)

epochs = args.epochs
best_accuracy = 0
kk = 0
prevloss = np.inf # Not used, can be removed if not needed for other parts

# Moving average configuration
window_size = 5
loss_history = deque(maxlen=window_size)
gradient_accumulation_steps = args.zero_grad_batch // args.batch_size
l1_lambda = args.l1_lambda

#  Training Loop
for epoch in range(epochs):
    # DataLoaders are created at the start of each epoch
    train_loader = create_dataloader(X, y, args.batch_size) # y still passes RMSD values
    val_loader = create_dataloader(X_val, y_val, args.batch_size)
    test_loader = create_dataloader(X_test, y_test, args.batch_size)
    CB513_loader = create_dataloader(CB513X, CB513Y, args.batch_size)

    loaders_dict={'val': val_loader, 'test':test_loader, 'CB513':CB513_loader, 'train':train_loader}
    
    if epoch < skip_epochs:
        print(f"Skipping epoch {epoch + 1}/{epochs}")
        kk += len(train_loader)
    else:
        model.train() # Set model to training mode
        epoch_losses = []

        for step, (Xt_batch, yt_batch_rmsd) in enumerate(train_loader):
            Xt_batch = Xt_batch.to(device)

            # Process target labels (yt_batch_rmsd) for CrossEntropyLoss:
            # It contains RMSD values, so we need to argmin to get the true class index.
            yt_batch_class_indices = torch.argmin(yt_batch_rmsd, dim=1).long().to(device)

            # Forward pass: model outputs raw logits
            y_pred = model(Xt_batch)
            loss = criterion(y_pred, yt_batch_class_indices) # Use the derived class indices here

            # L1 regularization
            l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            loss += l1_lambda * l1_penalty

            # Backward pass and gradient accumulation
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()    # Update model parameters
                optimizer.zero_grad() # Zero gradients for the next accumulation step

            epoch_losses.append(loss.item())
            kk += 1

        epoch_loss = np.mean(epoch_losses)
        loss_history.append(epoch_loss)
        moving_average_loss = sum(loss_history) / len(loss_history)
        lossmedian = statistics.median(epoch_losses)
        scheduler.step(moving_average_loss) # Update learning rate based on moving average loss

        print(f"Epoch: {epoch+1}/{epochs}, kk = {kk-1}, lossmedian = {lossmedian:.6f}, average loss: {epoch_loss:.6f}, learning rate = {optimizer.param_groups[0]['lr']:.7f}")

        # Validation and Accuracy Calculation
        N = args.e_acc
        if (epoch+1)%N==0 or epoch+1==epochs:
            # save_data_and_calculate_accuracy handles deriving class indices from RMSD values
            save_data_and_calculate_accuracy(model, optimizer, epoch, epoch_loss, loaders_dict, args, device, current_index)