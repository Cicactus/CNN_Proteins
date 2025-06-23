CNN_Q16.py - program to train CNN for protein local structure prediction by protein blocks without reliancy to the evolutionary information (i.e. sequence alignments).
training_script_CNN.sh - script for launching learning and iterating over hyperparameters.
prediction_Q16.py - program for calculating accuracies across datasets.
confusion_Q16.py - program for calculating confusion matrix and classification report.

1. Download datasets for learning:
   wget https://ftp.eimb.ru/Milch/Learn.Q16/learn_Q16.dbs.tar.gz
   or
   wget ftp://ftp.eimb.ru/Milch/Learn.Q16/learn_Q16.dbs.tar.gz

2. Extract the archive
    tar -xf learn_Q16.dbs.tar.gz

3. Install python3 and all proper dependencies:
   1. python 3 (tested with v. 3.8.10)
   2. numpy (tested with v. 1.22.3)
   3. PyTorch (tested with v. 1.12.1)
   4. `scikit-learn`, `matplotlib` and `seaborn` are also used for evaluation scripts.

3. If you want to learn from the scratch, you can choose to delete following files:
Q8_BEST_MODEL_7_5_3_2_1_512_tanh_leakyrelu_78.52_FOR_CPU
Q8_BEST_MODEL_7_5_3_2_1_512_tanh_leakyrelu_78.52
but the script will automatically delete them and save new ones if you train a model with higher validation accuracy.
The best models of each architecture configuration are saved to './trained_models/'.
   
4. Make the script execitable, modify the parameters, and run the script
    chmod +x ./training_script_CNN.sh
   ./training_script_CNN.sh