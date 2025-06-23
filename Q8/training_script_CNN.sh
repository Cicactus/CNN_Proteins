#!/bin/bash

# Default Python script
python_script="CNN_Q8.py"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --which_file) python_script="$2"; shift ;;   # Specify which script to run
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Parameters for the loop
learning_rates=(0.001)
batch_sizes=(2048) # Influences the performance
zero_grad_batch=(2048) # Influences the result
epochs=(100)
conv_layers=(7)
fc_layers=(5)
kernel_sizes=(3)
strides=(2)
paddings=(1)
conv_activations=("tanh")
fc_activations=("leaky_relu")
losses=("smooth_l1_loss")
first_layer_channels=(512)
l1_lambdas=(0.00001)
optimizers=("adam")

# Other parameters
accuracy_datasets=("val CB513") # Possible variants: val, test, CB513, train
E=10 # Calculating accuracy every E epochs
pretrained="n" # make argument Y if you want to further train a pre-trained model

# N value for how many instances of a loop to skip
N=0
counter=0

# Loop
for ep in "${epochs[@]}"; do
  for loss in "${losses[@]}"; do
    for bs in "${batch_sizes[@]}"; do
      for stride in "${strides[@]}"; do
        for lr in "${learning_rates[@]}"; do
          for zgb in "${zero_grad_batch[@]}"; do
            for ks in "${kernel_sizes[@]}"; do
              for flc in "${first_layer_channels[@]}"; do
                for fca in "${fc_activations[@]}"; do
                  for coa in "${conv_activations[@]}"; do
                    for fl in "${fc_layers[@]}"; do
                      for cl in "${conv_layers[@]}"; do
                        for pd in "${paddings[@]}"; do
                          for acc in "${accuracy_datasets[@]}"; do
                            for l1 in "${l1_lambdas[@]}"; do
                              for opt in "${optimizers[@]}"; do
                                # Skip the first N instances
                                if [[ $counter -ge $N ]]; then
                                  python3 "$python_script" \
                                    --lr_start "$lr" \
                                    --batch_size "$bs" \
                                    --epochs "$ep" \
                                    --conv_layers "$cl" \
                                    --fc_layers "$fl" \
                                    --kernel_size "$ks" \
                                    --stride "$stride" \
                                    --padding "$pd" \
                                    --fc_activation "$fca" \
                                    --conv_activation "$coa" \
                                    --loss "$loss" \
                                    --first_layer_channels "$flc" \
                                    --zero_grad_batch "$zgb" \
                                    --accuracy_datasets "$acc" \
                                    --e_acc "$E" \
                                    --pretrained "$pretrained" \
                                    --l1_lambda "$l1" \
                                    --optimizer "$opt"
                                fi
                                ((counter++))   # Increment the counter
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done