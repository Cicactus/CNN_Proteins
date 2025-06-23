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
from torch.utils.data import DataLoader, TensorDataset, Dataset
import sys

# --- Global Constants ---
LastModelName = "trained_models/Q8_LAST_MODEL"
BestModelName = "Q8_BEST_MODEL"

# --- Argument Parser ---
def get_arguments():
    parser = argparse.ArgumentParser(description="CNN Training Script for Q8")

    parser.add_argument("--lr_start", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument('--zero_grad_batch', type=int, default=2048, help="Effective batch size, determining gradient update frequency (must be >= batch size)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--conv_layers", type=int, default=2, help="Number of convolutional layers")
    parser.add_argument("--fc_layers", type=int, default=1, help="Number of fully connected layers")
    parser.add_argument("--kernel_size", type=int, default=10, help="Kernel size (for CNNs)")
    parser.add_argument('--stride', type=int, default=2, help="Stride value (for CNNs)")
    parser.add_argument('--padding', type=int, default=1, help="Padding value (for CNNs)")
    parser.add_argument('--conv_activation', type=str, default='tanh', help="Activation function for convolutional layers")
    parser.add_argument('--fc_activation', type=str, default='sigmoid', help="Activation function for fully connected layers")
    parser.add_argument('--first_layer_channels', type=int, default=32, help="Number of output channels for the first convolutional layer")
    parser.add_argument('--lr_method', type=str, default='', help="Learning rate reduction method")
    parser.add_argument('--accuracy_datasets', type=str, default='test CB513', help="Datasets on which accuracy is measured (space-separated)")
    parser.add_argument('--e_acc', type=int, default=10, help="Number of epochs after which accuracy is re-measured")
    parser.add_argument('--pretrained_model_path', type=str, default='',
                        help="Path to a pretrained model state_dict file (e.g., Q8_LAST_MODEL_..._FOR_CPU). If empty, no pretrained model is loaded.")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of parallel graphs (for GNNs) - kept for args consistency, but not used in CNN model")
    parser.add_argument('--l1_lambda', type=float, default=0.00001, help="L1 regularization lambda")
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer to use")
    parser.add_argument('--loss', type=str, default='cross_entropy_loss', help="Loss function (for logging)") # This is for logging purposes

    args = parser.parse_args()
    return args

# --- Dictionaries for Activations and Optimizers ---
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

# --- DataLoader Creation ---
def create_dataloader(X_data, y_data, batch_size):
    dataset = TensorDataset(torch.from_numpy(X_data), torch.from_numpy(y_data))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model Loading and Hyperparameter Check Helpers ---
def load_model_with_partial_weights(new_model, state_dict_path):
    old_state = torch.load(state_dict_path, map_location='cpu') # Load to CPU first
    new_state = new_model.state_dict()
    matched_state = {k: v for k, v in old_state.items() if k in new_state and v.size() == new_state[k].size()}
    new_state.update(matched_state)
    new_model.load_state_dict(new_state)
    skipped_layers = {k: v.size() for k, v in old_state.items() if k not in matched_state}
    return skipped_layers

def check_hyperparameter_match(old_state_dict, current_model_state_dict):
    for k in current_model_state_dict:
        if k in old_state_dict and old_state_dict[k].shape != current_model_state_dict[k].shape:
            return False
    return True

# --- Accuracy Calculation Functions ---
def calculate_classification_accuracy(y_pred_logits, true_labels_class_indices):
    predicted_classes = torch.argmax(y_pred_logits, dim=1)
    correct_predictions = (predicted_classes == true_labels_class_indices).sum().item()
    total_samples = true_labels_class_indices.size(0)
    if total_samples == 0:
        return 0.0
    return correct_predictions / total_samples

def get_loader_accuracy_CNN(loader, model, device):
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        model.eval()
        for Xt_batch, yt_batch_labels in loader:
            Xt_batch = Xt_batch.to(device)
            # True class indices are derived by argmaxing the one-hot like labels
            true_labels_class_indices = torch.argmax(yt_batch_labels, dim=1).long().to(device)

            y_pred_logits = model(Xt_batch) # Model now outputs raw logits

            total_correct += (torch.argmax(y_pred_logits, dim=1) == true_labels_class_indices).sum().item()
            total_samples += true_labels_class_indices.size(0)
    
    if total_samples == 0:
        return 0.0
    return total_correct / total_samples

def get_all_accuracy_results(model, device, loaders, accuracy_datasets):
    with torch.no_grad():
        model.eval()
        dataset_names = accuracy_datasets.split()
        accuracies = {}
        for dataset_name in dataset_names:
            if dataset_name in loaders:
                loader = loaders[dataset_name]
                acc = get_loader_accuracy_CNN(loader, model, device)
                accuracies[dataset_name] = acc
                print(f"{dataset_name} Accuracy: {acc * 100:.2f}%")
            else:
                print(f"Loader for dataset {dataset_name} not found!")
        return accuracies

# --- File Operations Helpers ---
def delete_file(file_path):
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def find_and_load_q8_index_params(model_file_path, model_prefix="Q8"):
    file_name = Path(model_file_path).name

    q8_model_regex = re.compile(
        rf"^{re.escape(model_prefix)}_([A-Z_]+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_([a-zA-Z]+)_([a-zA-Z]+)_(\d+\.\d+)(?:_FOR_CPU)?$"
    )

    match = q8_model_regex.match(file_name)
    if match:
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
        return (None, None, None, None, None, None, None, None, None)

def extract_cnn_model_info(folder_path, model_prefix):
    folder = Path(folder_path)
    model_info_list = []

    model_regex = re.compile(
        rf"^{re.escape(model_prefix)}_([A-Z_]+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_([a-zA-Z]+)_([a-zA-Z]+)_(\d+\.\d+)(?:_FOR_CPU)?$"
    )

    for file_path in folder.glob(f"{model_prefix}*"):
        if file_path.is_file() and "GNN" not in file_path.name:
            match = model_regex.match(file_path.name)
            if match:
                conv_l, fc_l, k_size, strd, pad, f_ch = map(int, match.groups()[1:7])
                conv_act, fc_act = match.groups()[7:9]
                model_accuracy = float(match.group(9))

                current_index_str = f'{conv_l}_{fc_l}_{k_size}_{strd}_{pad}_{f_ch}_{conv_act}_{fc_act}'
                
                model_info_list.append((file_path, current_index_str, model_accuracy))
    return model_info_list

# --- Main Save and Accuracy Function ---
def save_data_and_calculate_accuracy(
    model, optimizer, epoch, epoch_loss, loaders, args, device, current_index_string
):
    csv_file = "training_results.csv"
    Path("trained_models").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model.eval()

        accuracies = get_all_accuracy_results(
            model, device, loaders, args.accuracy_datasets
        )

        test_accuracy = accuracies.get('test', 0)
        val_accuracy = accuracies.get('val', 0)
        CB513_accuracy = accuracies.get('CB513', 0)

        data = {
            "lr_start": args.lr_start, "batch_size": args.batch_size, "zero_grad_batch": args.zero_grad_batch,
            "epochs": epoch + 1, "conv_layers": args.conv_layers, "fc_layers": args.fc_layers,
            "kernel_size": args.kernel_size, "stride": args.stride, "padding": args.padding,
            "conv_activation": args.conv_activation, "fc_activation": args.fc_activation,
            "loss_function": args.loss, # This is the string from argparse, not the loss value
            "first_layer_channels": args.first_layer_channels, "lr_final": optimizer.param_groups[0]['lr'],
            "lr_method": args.lr_method, "final_loss": epoch_loss, "test_accuracy": test_accuracy,
            "validation_accuracy": val_accuracy, "CB513_accuracy": CB513_accuracy,
            "optimizer": args.optimizer, "l1_lambda": args.l1_lambda, "pretrained_model_path": args.pretrained_model_path,
            "accuracy_datasets": args.accuracy_datasets
        }

        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

        last_model_base_name = f'Q8_LAST_MODEL_{current_index_string}_{val_accuracy * 100:.2f}'

        existing_last_model_path = None
        existing_last_model_accuracy = -1.0

        trained_models_in_folder = extract_cnn_model_info("trained_models", "Q8_LAST_MODEL") # Adjust prefix for search
        for file_path, parsed_index_str, accuracy_from_filename in trained_models_in_folder:
            if parsed_index_str == current_index_string:
                existing_last_model_path = file_path
                existing_last_model_accuracy = accuracy_from_filename
                break

        if val_accuracy * 100 > existing_last_model_accuracy:
            if existing_last_model_path:
                delete_file(existing_last_model_path)
                # Also delete the _FOR_CPU version if it exists
                cpu_path_to_delete = existing_last_model_path.parent / f"{existing_last_model_path.stem}_FOR_CPU"
                delete_file(cpu_path_to_delete)
                print(f"Removed previous 'last' model for index {current_index_string} (Accuracy: {existing_last_model_accuracy:.2f}%).")

            # Save the new last model
            torch.save(model, f'trained_models/{last_model_base_name}')
            torch.save(model.state_dict(), f'trained_models/{last_model_base_name}_FOR_CPU')
            print(f"Saved current model as the new 'last' model for index {current_index_string} (Validation Accuracy: {val_accuracy * 100:.2f}%).")
        else:
            print(f"Current model's accuracy ({val_accuracy * 100:.2f}%) is not better than the existing 'last' model for index {current_index_string} ({existing_last_model_accuracy:.2f}%). No update to 'last' model.")

        overall_best_accuracy = -1.0
        overall_best_model_path = None

        best_models_in_folder = extract_cnn_model_info(".", "Q8_BEST_MODEL")
        for file_path, _, accuracy_from_filename in best_models_in_folder:
            if accuracy_from_filename > overall_best_accuracy:
                overall_best_accuracy = accuracy_from_filename
                overall_best_model_path = file_path
        
        best_model_base_name = f'Q8_BEST_MODEL_{current_index_string}_{val_accuracy * 100:.2f}'

        if val_accuracy * 100 > overall_best_accuracy:
            if overall_best_model_path:
                delete_file(overall_best_model_path)
                # Also delete the _FOR_CPU version if it exists
                cpu_path_to_delete = overall_best_model_path.parent / f"{overall_best_model_path.stem}_FOR_CPU"
                delete_file(cpu_path_to_delete)
            
            torch.save(model, f'{best_model_base_name}')
            torch.save(model.state_dict(), f'{best_model_base_name}_FOR_CPU')
            print(f"Saved current model as the **overall best model** (Validation Accuracy: {val_accuracy * 100:.2f}%).")
        else:
            print(f"Current model's accuracy ({val_accuracy * 100:.2f}%) is not better than the overall best model ({overall_best_accuracy:.2f}%).")
    
    return True

# Model Definition
class MultiOutputRegression(nn.Module):
    def __init__(self, conv_layers, fc_layers, kernel_size, stride, padding,
                 first_layer_channels, conv_activation_name, fc_activation_name,
                 num_classes, input_len, freeze_conv=False):
        super().__init__()

        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.first_layer_channels = first_layer_channels
        self.num_classes = num_classes
        self.input_len = input_len

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.conv_activation = activations[conv_activation_name]
        self.fc_activation = activations[fc_activation_name]

        l = self.input_len
        for i in range(self.conv_layers):
            channels_this_layer = self.first_layer_channels
            in_channels = 1 if i == 0 else channels_this_layer
            conv = nn.Conv1d(in_channels, channels_this_layer,
                             kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            bn = nn.BatchNorm1d(channels_this_layer)
            self.convs.append(conv)
            self.bns.append(bn)
            l = int((l + 2 * self.padding - self.kernel_size) / self.stride + 1)

        flat_dim = int(l) * self.first_layer_channels
        if flat_dim <= 1:
            raise ValueError(f"Invalid flattened dimension after convolution layers: {flat_dim}. Adjust conv_layers or kernel_size.")

        powers = np.linspace(int(np.floor(np.log2(flat_dim - 1))), int(np.ceil(np.log2(self.num_classes + 1))), self.fc_layers - 1)
        values = [flat_dim] + [int(2 ** p) for p in powers] + [self.num_classes]

        for i in range(self.fc_layers):
            fc = nn.Linear(values[i], values[i + 1])
            self.fcs.append(fc)

        if freeze_conv:
            for conv_layer in self.convs:
                for param in conv_layer.parameters():
                    param.requires_grad = False
            for bn_layer in self.bns:
                for param in bn_layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        for i in range(self.conv_layers):
            x = self.conv_activation(self.bns[i](self.convs[i](x)))
        x = x.flatten(start_dim=1)
        for i in range(len(self.fcs) - 1):
            x = self.fc_activation(self.fcs[i](x))
        return self.fcs[-1](x)


# --- Main Execution Block ---
if __name__ == "__main__":
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

    current_index_string = f'{args.conv_layers}_{args.fc_layers}_{args.kernel_size}_{args.stride}_{args.padding}_{args.first_layer_channels}_{args.conv_activation}_{args.fc_activation}'
    print(f"Current model index string: {current_index_string}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    db_folder_name='learn_Q8'

    try:
        X = np.load(f"./{db_folder_name}/train_Q8X.npy").astype('float32')
        y = np.load(f"./{db_folder_name}/train_Q8Y.npy").astype('float32')
        CB513X = np.load(f"./{db_folder_name}/CB513X.npy").astype('float32') # Corrected path
        CB513Y = np.load(f"./{db_folder_name}/CB513Y.npy").astype('float32') # Corrected path
        X_test = np.load(f"./{db_folder_name}/test_for_dsspX.npy").astype('float32')
        y_test = np.load(f"./{db_folder_name}/test_for_dsspY.npy").astype('float32')
        X_val = np.load(f"./{db_folder_name}/validation_for_dsspX.npy").astype('float32')
        y_val = np.load(f"./{db_folder_name}/validation_for_dsspY.npy").astype('float32')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}. Please ensure all .npy files are in the correct directories.")
        sys.exit(1)

    X = X.reshape(X.shape[0], 1, X.shape[-1])
    CB513X = CB513X.reshape(CB513X.shape[0], 1, CB513X.shape[-1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[-1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[-1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input sequence length: {X.shape[-1]}")
    print(f"Number of classes (from y.shape[-1]): {y.shape[-1]}")

    # Pretrained model handling (initial instantiation for checking)
    model = MultiOutputRegression(
        conv_layers=args.conv_layers,
        fc_layers=args.fc_layers,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        first_layer_channels=args.first_layer_channels,
        conv_activation_name=args.conv_activation,
        fc_activation_name=args.fc_activation,
        num_classes=y.shape[-1],
        input_len=X.shape[-1]
    ).to(device)

    skip_epochs = 0
    if args.pretrained_model_path:
        print(f"Attempting to load pretrained model from: {args.pretrained_model_path}")
        if os.path.exists(args.pretrained_model_path):
            old_state = torch.load(args.pretrained_model_path, map_location='cpu') # Load to CPU for checking
            temp_model_for_check = MultiOutputRegression(
                conv_layers=args.conv_layers, fc_layers=args.fc_layers, kernel_size=args.kernel_size,
                stride=args.stride, padding=args.padding, first_layer_channels=args.first_layer_channels,
                conv_activation_name=args.conv_activation, fc_activation_name=args.fc_activation,
                num_classes=y.shape[-1], input_len=X.shape[-1]
            ).to('cpu') # Create on CPU for checking
            temp_state = temp_model_for_check.state_dict()
            
            hyperparameters_match = check_hyperparameter_match(old_state, temp_state)

            model = MultiOutputRegression(
                conv_layers=args.conv_layers,
                fc_layers=args.fc_layers,
                kernel_size=args.kernel_size,
                stride=args.stride,
                padding=args.padding,
                first_layer_channels=args.first_layer_channels,
                conv_activation_name=args.conv_activation,
                fc_activation_name=args.fc_activation,
                num_classes=y.shape[-1],
                input_len=X.shape[-1],
                freeze_conv=not hyperparameters_match # Freeze if hyperparameters don't match
            ).to(device)

            if hyperparameters_match:
                print("Hyperparameters match! All layers will remain unfrozen.")
            else:
                print("Hyperparameters do not match. Convolutional layers will be frozen.")

            skipped_layers = load_model_with_partial_weights(model, args.pretrained_model_path)
            print(f"Skipped layers due to mismatched sizes (pretrained model): {skipped_layers}")

            while True:
                try:
                    skip_epochs = int(input('How many epochs to skip (from previous training)? '))
                    if skip_epochs >= 0:
                        break
                    else:
                        print("Please enter a non-negative integer.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print(f"Warning: Pretrained model not found at: {args.pretrained_model_path}. Starting training from scratch.")
            skip_epochs = 0
    else:
        print("No pretrained model specified. Starting training from scratch.")


    num_classes = y.shape[-1]

    y_train_indices_for_counts = np.argmax(y, axis=1).astype(int)

    class_counts = Counter(y_train_indices_for_counts)

    weights = torch.tensor([1.0 / (class_counts.get(i, 0) if class_counts.get(i, 0) > 0 else 1e-6) for i in range(num_classes)],
                            device=device, dtype=torch.float)

    weights = weights / weights.sum() * num_classes

    print(f"Calculated class counts: {class_counts}")
    print(f"Calculated class weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)

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
    
    # Moving average configuration
    window_size = 5
    loss_history = deque(maxlen=window_size)
    gradient_accumulation_steps = args.zero_grad_batch // args.batch_size
    l1_lambda = args.l1_lambda

    for epoch in range(epochs):
        train_loader = create_dataloader(X, y, args.batch_size)
        val_loader = create_dataloader(X_val, y_val, args.batch_size)
        test_loader = create_dataloader(X_test, y_test, args.batch_size)
        CB513_loader = create_dataloader(CB513X, CB513Y, args.batch_size)

        loaders_dict={'val': val_loader, 'test':test_loader, 'CB513':CB513_loader, 'train':train_loader}
        
        if epoch < skip_epochs:
            print(f"Skipping epoch {epoch + 1}/{epochs} (due to pretrained model setting).")
            kk += len(train_loader) 
            with torch.no_grad():
                # Step the scheduler even if skipping, to simulate loss progression
                # For `ReduceLROnPlateau`, passing a large value like `np.inf` will prevent reduction
                # but allow the `patience` counter to correctly increment for a flat loss.
                scheduler.step(np.inf) 
        else:
            model.train()
            epoch_losses = []

            optimizer.zero_grad() 

            for step, (Xt_batch, yt_batch_labels) in enumerate(train_loader):
                Xt_batch = Xt_batch.to(device)
                
                # This is correct: convert one-hot or multi-hot labels to class indices
                yt_batch_class_indices = torch.argmax(yt_batch_labels, dim=1).long().to(device)

                # Model now outputs raw logits
                y_pred_logits = model(Xt_batch) 
                
                # Use the defined 'criterion' (nn.CrossEntropyLoss)
                loss = criterion(y_pred_logits, yt_batch_class_indices) 

                l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                loss += l1_lambda * l1_penalty

                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_losses.append(loss.item())
                kk += 1

            epoch_loss = np.mean(epoch_losses)
            loss_history.append(epoch_loss)
            moving_average_loss = sum(loss_history) / len(loss_history)
            lossmedian = statistics.median(epoch_losses)
            scheduler.step(moving_average_loss)

            print(f"Epoch: {epoch+1}/{epochs}, kk = {kk-1}, lossmedian = {lossmedian:.6f}, average loss: {epoch_loss:.6f}, learning rate = {optimizer.param_groups[0]['lr']:.7f}")

            N = args.e_acc
            if (epoch+1)%N==0 or epoch+1==epochs:
                save_data_and_calculate_accuracy(model, optimizer, epoch, epoch_loss, loaders_dict, args, device, current_index_string)

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)