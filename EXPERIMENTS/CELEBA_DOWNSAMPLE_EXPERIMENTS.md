ERM on CelebA Downsample:
`python generate_downstream.py --exp_name CelebA_SMALL_TEST --dataset CelebA --method ERM --lr 1e-5 --weight_decay 0.1 --downsample`
`bash results/CelebA/CelebA_SMALL_TEST/ERM_upweight_0_epochs_50_lr_0.001_weight_decay_0.0001/job.sh`
`python analysis.py --exp_name CelebA_SMALL_TEST --dataset CelebA --exp_substring ERM_upweight_0_epochs_50_lr_0.001_weight_decay_0.0001`

Expected Results:
Val Robust Worst Group val   acc (early stop at epoch 44): 41.2
Val Robust Worst Group test  acc (early stop at epoch 44): 45.6
Val Average Acc val   acc (early stop at epoch 44): 93.0
Val Average Acc test  acc (early stop at epoch 44): 93.6
group 0 acc val   acc (early stop at epoch 44): 91.5
group 0 acc test  acc (early stop at epoch 44): 93.4
group 1 acc val   acc (early stop at epoch 44): 98.3
group 1 acc test  acc (early stop at epoch 44): 98.1
group 2 acc val   acc (early stop at epoch 44): 85.3
group 2 acc test  acc (early stop at epoch 44): 83.9
group 3 acc val   acc (early stop at epoch 44): 41.2
group 3 acc test  acc (early stop at epoch 44): 45.6

Upweight Misclassified:
`python process_training.py --exp_name CelebA_SMALL_TEST --dataset CelebA --folder_name ERM_upweight_0_epochs_50_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --upweight_misclassified --batch_size 32 --downsample`

`bash results/CelebA/CelebA_SMALL_TEST/train_downstream_ERM_upweight_0_epochs_50_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_50_lr_0.001_weight_decay_0.0001_REWEIGHT/job.sh`

`python analysis.py --exp_name CelebA_SMALL_TEST/train_downstream_ERM_upweight_0_epochs_50_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset CelebA --exp_substring JTT_upweight_50_epochs_50_lr_0.001_weight_decay_0.0001_REWEIGHT`