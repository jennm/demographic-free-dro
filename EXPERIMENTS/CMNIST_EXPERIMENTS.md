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

JTT DRO:
`python process_training.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --folder_name ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001 --lr 0.001 --weight_decay 0.0001 --deploy --final_epoch 1 --batch_size 32 --jtt_fake_dro`

`bash results/ColoredMNIST/ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_JTT_FAKE_DRO/job.sh`

`python analysis.py --exp_name ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/ --dataset ColoredMNIST --exp_substring JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_JTT_FAKE_DRO`

Val Robust Worst Group val   acc (early stop at epoch 4): 52.8
Val Robust Worst Group test  acc (early stop at epoch 4): 53.5
Val Average Acc val   acc (early stop at epoch 4): 79.1
Val Average Acc test  acc (early stop at epoch 4): 79.5
group 0 acc val   acc (early stop at epoch 4): 86.0
group 0 acc test  acc (early stop at epoch 4): 85.5
group 1 acc val   acc (early stop at epoch 4): 85.1
group 1 acc test  acc (early stop at epoch 4): 86.3
group 2 acc val   acc (early stop at epoch 4): 52.8
group 2 acc test  acc (early stop at epoch 4): 53.5
group 3 acc val   acc (early stop at epoch 4): 91.5
group 3 acc test  acc (early stop at epoch 4): 92.3

Our Algorithm:
`python generate_downstream.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --method ERM --batch_size 32`
`python run_expt.py -s confounder -d ColoredMNIST -t target -c confounder --batch_size 32 --root_dir ./ --n_epochs 5 --aug_col None --log_dir results/ColoredMNIST/ColoredMNIST_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/ColoredMNIST/ColoredMNIST_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type group_dro --classifier_groups True --group_info_path groups_from_classifiers_info.pt`

`python run_expt.py -s confounder -d ColoredMNIST -t target -c confounder --batch_size 32 --root_dir ./ --n_epochs 5 --aug_col None --log_dir results/ColoredMNIST/ColoredMNIST_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/ColoredMNIST/ColoredMNIST_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type group_dro --classifier_groups True --group_info_path groups_from_classifiers_info.pt`
`python analysis.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --exp_substring ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001_CLSGROUPS`

Results:
Val Robust Worst Group val   acc (early stop at epoch 4): 85.8
Val Robust Worst Group test  acc (early stop at epoch 4): 96.4
Val Average Acc val   acc (early stop at epoch 4): 85.9
Val Average Acc test  acc (early stop at epoch 4): 96.5
group 0 acc val   acc (early stop at epoch 4): 85.9
group 0 acc test  acc (early stop at epoch 4): 96.4
group 1 acc val   acc (early stop at epoch 4): 85.9
group 1 acc test  acc (early stop at epoch 4): 96.5
group 2 acc val   acc (early stop at epoch 4): 85.9
group 2 acc test  acc (early stop at epoch 4): 96.5
group 3 acc val   acc (early stop at epoch 4): 85.8
group 3 acc test  acc (early stop at epoch 4): 96.5

python generate_downstream.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --method ERM --batch_size 32 --classifier_group_path test
bash results/ColoredMNIST/ColoredMNIST_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001_CLSGROUPS/job.sh
python analysis.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --exp_substring ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001_CLSGROUPS


python generate_downstream.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --method GROUP_DRO --batch_size 32
bash results/ColoredMNIST/ColoredMNIST_TEST/GROUP_DRO_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/job.sh

python analysis.py --exp_name ColoredMNIST_TEST --dataset ColoredMNIST --exp_substring GROUP_DRO_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001

[0, 3, 0, 3, 3, 3, 2, 0, 2, 3, 3, 0, 0, 2, 3, 2, 1, 2, 3, 1, 0, 1, 2, 0,
        1, 2, 0, 2, 2, 0, 2, 0]