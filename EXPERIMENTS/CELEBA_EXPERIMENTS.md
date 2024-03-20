ERM on CelebA:
`python generate_downstream.py --exp_name CelebA_TEST --dataset CelebA --method ERM --lr 1e-5 --weight_decay 0.1`
`bash results/CelebA/CelebA_TEST/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/job.sh`
`python analysis.py --exp_name CelebA_TEST --dataset CelebA --exp_substring ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1`

Val Robust Worst Group val   acc (early stop at epoch 45): 42.9
Val Robust Worst Group test  acc (early stop at epoch 45): 45.0
Val Average Acc val   acc (early stop at epoch 45): 95.6
Val Average Acc test  acc (early stop at epoch 45): 95.8
group 0 acc val   acc (early stop at epoch 45): 94.3
group 0 acc test  acc (early stop at epoch 45): 95.3
group 1 acc val   acc (early stop at epoch 45): 99.6
group 1 acc test  acc (early stop at epoch 45): 99.4
group 2 acc val   acc (early stop at epoch 45): 91.2
group 2 acc test  acc (early stop at epoch 45): 90.3
group 3 acc val   acc (early stop at epoch 45): 42.9
group 3 acc test  acc (early stop at epoch 45): 45.0

Upweight Misclassified:
`python process_training.py --exp_name CelebA_TEST --dataset CelebA --folder_name ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1 --lr 1e-5 --weight_decay 0.1 --deploy --final_epoch 1 --upweight_misclassified`

`bash results/CelebA/CelebA_TEST/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1_REWEIGHT/job.sh`

`python analysis.py --exp_name CelebA_TEST/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/ --dataset ColoredMNIST --exp_substring JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1_REWEIGHT`