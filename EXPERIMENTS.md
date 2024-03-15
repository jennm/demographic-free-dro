To generate colored mnist:
`python write_colored_mnist.py`

ERM on colored mnist:
`python generate_downstream.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --method ERM --batch_size 32`
`bash results/ColoredMNIST/ColoredMNIST_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/job.sh`
`python analysis.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --exp_substring ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001`

Expected Results:
Val Robust Worst Group val   acc (early stop at epoch 3): 60.6
Val Robust Worst Group test  acc (early stop at epoch 3): 62.1
Val Average Acc val   acc (early stop at epoch 3): 82.5
Val Average Acc test  acc (early stop at epoch 3): 84.3
group 0 acc val   acc (early stop at epoch 3): 76.4
group 0 acc test  acc (early stop at epoch 3): 79.1
group 1 acc val   acc (early stop at epoch 3): 97.8
group 1 acc test  acc (early stop at epoch 3): 98.2
group 2 acc val   acc (early stop at epoch 3): 96.0
group 2 acc test  acc (early stop at epoch 3): 96.4
group 3 acc val   acc (early stop at epoch 3): 60.6
group 3 acc test  acc (early stop at epoch 3): 62.1

JTT on colored mnist:
`python process_training.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --folder_name ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --batch_size 32`
`bash results/ColoredMNIST/ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001/job.sh`
`python analysis.py --exp_name ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset ColoredMNIST --exp_substring JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001`

Expected Results:
Val Robust Worst Group val   acc (early stop at epoch 1): 94.2
Val Robust Worst Group test  acc (early stop at epoch 1): 93.9
Val Average Acc val   acc (early stop at epoch 1): 95.8
Val Average Acc test  acc (early stop at epoch 1): 96.0
group 0 acc val   acc (early stop at epoch 1): 95.1
group 0 acc test  acc (early stop at epoch 1): 96.1
group 1 acc val   acc (early stop at epoch 1): 97.1
group 1 acc test  acc (early stop at epoch 1): 97.0
group 2 acc val   acc (early stop at epoch 1): 96.8
group 2 acc test  acc (early stop at epoch 1): 96.9
group 3 acc val   acc (early stop at epoch 1): 94.2
group 3 acc test  acc (early stop at epoch 1): 93.9

Lambda Loss on colored mnist:
`python process_training.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --folder_name ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --lambda_loss --batch_size 32`
`bash results/ColoredMNIST/ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_LAMBDA_LOSS/job.sh`
`python analysis.py --exp_name ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset ColoredMNIST --exp_substring JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_LAMBDA_LOSS`

Expected Results:
Val Robust Worst Group val   acc (early stop at epoch 4): 87.4
Val Robust Worst Group test  acc (early stop at epoch 4): 84.5
Val Average Acc val   acc (early stop at epoch 4): 93.6
Val Average Acc test  acc (early stop at epoch 4): 93.0
group 0 acc val   acc (early stop at epoch 4): 97.9
group 0 acc test  acc (early stop at epoch 4): 98.3
group 1 acc val   acc (early stop at epoch 4): 93.0
group 1 acc test  acc (early stop at epoch 4): 93.8
group 2 acc val   acc (early stop at epoch 4): 87.4
group 2 acc test  acc (early stop at epoch 4): 84.5
group 3 acc val   acc (early stop at epoch 4): 95.7
group 3 acc test  acc (early stop at epoch 4): 95.4

Upweight Misclassified:
`python process_training.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --folder_name ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --upweight_misclassified --batch_size 32`
`bash results/ColoredMNIST/ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_REWEIGHT/job.sh`
`python analysis.py --exp_name ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset ColoredMNIST --exp_substring JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_REWEIGHT`

Expected Results:
Val Robust Worst Group val   acc (early stop at epoch 2): 93.6
Val Robust Worst Group test  acc (early stop at epoch 2): 92.9
Val Average Acc val   acc (early stop at epoch 2): 94.7
Val Average Acc test  acc (early stop at epoch 2): 94.8
group 0 acc val   acc (early stop at epoch 2): 94.0
group 0 acc test  acc (early stop at epoch 2): 95.4
group 1 acc val   acc (early stop at epoch 2): 95.2
group 1 acc test  acc (early stop at epoch 2): 95.4
group 2 acc val   acc (early stop at epoch 2): 95.9
group 2 acc test  acc (early stop at epoch 2): 95.2
group 3 acc val   acc (early stop at epoch 2): 93.6
group 3 acc test  acc (early stop at epoch 2): 92.9