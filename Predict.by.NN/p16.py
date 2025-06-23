#!/usr/bin/python3
#
# script to perform protein secondary structure prediction by precomputed NN model
# (C) Yuri V. Milchevskiy, Yuri V. Kravatsky
# email: milch@eimb.ru
#
# Dependencies:
#  1. python 3 (tested with python 3.8.10)
#  2. numpy (tested with 1.22.3)
#  3. pandas (tested with 1.5.1)
#  4. PyTorch (tested with 1.12.1)
#
################################################################################################################
################################# PREDICTION USING A PRE-MADE MODEL "REAL_DATA_MODEL" ##########################
################################################################################################################ 
import numpy as np
import torch
import torch.nn as nn
import re
import sys
import pandas as pd
from pathlib import Path

# --- Function to find and load model parameters from filename ---
def find_and_load_q16_index_params(folder_path=".", model_prefix="Q16"):
    """
    Finds the first model that starts with 'Q16' in the specified folder,
    extracts its index values from the filename, and returns them along with the full path to the model file.
    """
    folder = Path(folder_path)

    # Regex to capture all parts of the Q16 model filename
    q16_model_regex = re.compile(
        rf"^{re.escape(model_prefix)}_([A-Z_]+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_([a-zA-Z]+)_([a-zA-Z]+)_(\d+\.\d+)(?:_FOR_CPU)?$"
    )

    for file_path in folder.iterdir():
        # Check if it's a file and starts with the prefix
        if file_path.is_file() and file_path.name.startswith(model_prefix):
            match = q16_model_regex.match(file_path.name)
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

                # Return the first match found, including its path
                return (conv_layers, fc_layers, kernel_size, stride, padding,
                        first_layer_channels, conv_activation, fc_activation, accuracy, str(file_path))

    return (None, None, None, None, None, None, None, None, None, None) # Return Nones if not found


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


def printf(format, *args):
    sys.stdout.write(format % args)

def accuracy(yt, y_pred):
    n_correct = 0; n_wrong = 0
    for i in range(len(yt)):
        t_index=torch.argmin(yt[i])
        pred_index=torch.argmax(y_pred[i])

        if t_index == pred_index:
            n_correct += 1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc

def get_predicted_PB_set(y_pred):
    PB='abcdefghijklmnop'
    PB_set=''

    for i in range(len(y_pred)):
        pred_index=torch.argmax(y_pred[i])
        PB_set += PB[pred_index]

    return PB_set

# Model definition
class Model(nn.Module):
    def __init__(self, conv_layers, fc_layers, kernel_size, stride, padding,
                 first_layer_channels, conv_activation_name='tanh',
                 fc_activation_name='leakyrelu', n_outputs=16):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.conv_activation = activations[conv_activation_name]
        self.fc_activation = activations[fc_activation_name]
        length = 380
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
                             f"It must be greater than 1 to correctly determine FC layer dimensions. "
                             f"Adjust conv_layers, kernel_size, stride, or padding.")

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

# Find the model parameters and its file path.
(conv_layers, fc_layers, kernel_size, stride, padding,
 first_layer_channels, conv_activation, fc_activation, accuracy, model_file_to_load) = \
    find_and_load_q16_index_params(folder_path=".") # Adjust folder_path as needed

# Check if parameters and model path were successfully loaded
if conv_layers is None or model_file_to_load is None:
    print("Error: Could not find or parse Q16 model parameters/file from filename. Exiting.")
    sys.exit(1)

# Instantiate the model using the loaded parameters
# This is the part that was missing or incorrectly placed!
model = Model(
    conv_layers=conv_layers,
    fc_layers=fc_layers,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    first_layer_channels=first_layer_channels,
    conv_activation_name=conv_activation,
    fc_activation_name=fc_activation,
    n_outputs=16 # Assuming n_outputs is always 16 for Q16 models
)

dev = "cpu"
device = torch.device(dev)

model.load_state_dict(torch.load(model_file_to_load, map_location=device))
model = model.to(device)

model.eval() # Set model to evaluation mode

name = sys.argv[1]

Xname=name

dfX = pd.read_csv(Xname, header=None,delimiter='\t')

X = dfX.values[:, :]

# ensure input data is floats
X = X.astype('float16')

Xt=torch.from_numpy(X).float().to(device)
Xt = Xt.unsqueeze(1) # This adds the channel dimension (1)
del X

y_pred = model(Xt)

length = Xt.size(dim=0) # This 'length' is actually batch_size here

PB_set=get_predicted_PB_set(y_pred)

print(PB_set)