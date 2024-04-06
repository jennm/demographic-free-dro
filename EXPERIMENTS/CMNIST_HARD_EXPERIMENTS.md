To generate colored mnist:
`python write_colored_mnist_hard.py`

ERM on colored mnist:
`python generate_downstream.py --exp_name ColoredMNIST_HARD_TEST --dataset ColoredMNIST_HARD --method ERM --batch_size 32`
`bash results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/job.sh`
`python analysis.py --exp_name ColoredMNIST_HARD_TEST --dataset ColoredMNIST_HARD --exp_substring ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001`

Val Robust Worst Group val   acc (early stop at epoch 4): 42.1
Val Robust Worst Group test  acc (early stop at epoch 4): 43.0
Val Average Acc val   acc (early stop at epoch 4): 85.0
Val Average Acc test  acc (early stop at epoch 4): 85.0
group 0 acc val   acc (early stop at epoch 4): 89.0
group 0 acc test  acc (early stop at epoch 4): 88.8
group 1 acc val   acc (early stop at epoch 4): 99.6
group 1 acc test  acc (early stop at epoch 4): 99.6
group 2 acc val   acc (early stop at epoch 4): 63.4
group 2 acc test  acc (early stop at epoch 4): 66.6
group 3 acc val   acc (early stop at epoch 4): 42.1
group 3 acc test  acc (early stop at epoch 4): 43.0

JTT on colored mnist:
`python process_training.py --exp_name ColoredMNIST_HARD_TEST --dataset ColoredMNIST_HARD --folder_name ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --batch_size 32`
`bash results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001/job.sh`
`python analysis.py --exp_name ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset ColoredMNIST_HARD --exp_substring JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001`


Lambda Loss on colored mnist:
`python process_training.py --exp_name ColoredMNIST_HARD_TEST --dataset ColoredMNIST_HARD --folder_name ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --lambda_loss --batch_size 32`
`bash results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_LAMBDA_LOSS/job.sh`
`python analysis.py --exp_name ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset ColoredMNIST_HARD --exp_substring JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_LAMBDA_LOSS`


Upweight Misclassified:
`python process_training.py --exp_name ColoredMNIST_HARD_TEST --dataset ColoredMNIST_HARD --folder_name ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --upweight_misclassified --batch_size 32`
`bash results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_REWEIGHT/job.sh`
`python analysis.py --exp_name ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset ColoredMNIST_HARD --exp_substring JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_REWEIGHT`

Val Robust Worst Group val   acc (early stop at epoch 4): 92.9
Val Robust Worst Group test  acc (early stop at epoch 4): 93.1
Val Average Acc val   acc (early stop at epoch 4): 96.5
Val Average Acc test  acc (early stop at epoch 4): 96.6
group 0 acc val   acc (early stop at epoch 4): 97.5
group 0 acc test  acc (early stop at epoch 4): 97.5
group 1 acc val   acc (early stop at epoch 4): 96.2
group 1 acc test  acc (early stop at epoch 4): 97.1
group 2 acc val   acc (early stop at epoch 4): 92.9
group 2 acc test  acc (early stop at epoch 4): 93.1
group 3 acc val   acc (early stop at epoch 4): 95.8
group 3 acc test  acc (early stop at epoch 4): 94.8

JTT DRO:
`python process_training.py --exp_name ColoredMNIST_HARD_TEST --dataset ColoredMNIST_HARD --folder_name ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --batch_size 32 --jtt_fake_dro`

`bash results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_JTT_FAKE_DRO/job.sh`

`python analysis.py --exp_name ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset ColoredMNIST_HARD --exp_substring JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_JTT_FAKE_DRO`

Val Robust Worst Group val   acc (early stop at epoch 4): 94.0
Val Robust Worst Group test  acc (early stop at epoch 4): 94.4
Val Average Acc val   acc (early stop at epoch 4): 96.1
Val Average Acc test  acc (early stop at epoch 4): 96.2
group 0 acc val   acc (early stop at epoch 4): 96.6
group 0 acc test  acc (early stop at epoch 4): 96.7
group 1 acc val   acc (early stop at epoch 4): 95.8
group 1 acc test  acc (early stop at epoch 4): 95.6
group 2 acc val   acc (early stop at epoch 4): 95.0
group 2 acc test  acc (early stop at epoch 4): 95.1
group 3 acc val   acc (early stop at epoch 4): 94.0
group 3 acc test  acc (early stop at epoch 4): 94.4