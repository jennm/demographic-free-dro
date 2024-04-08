TODO:

3) test out resume keyword
4) get full celebA jtt
6) custom algo



python groups.py --log_dir results/CelebA/CelebA_TEST/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/model_outputs --layer_nums 3


Create a new temp dataset object where we compute embeddings on the fly

python run_expt.py -s confounder -d ColoredMNIST -t target -c confounder --batch_size 32 --root_dir ./ --n_epochs 5 --aug_col None --log_dir results/ColoredMNIST/ColoredMNIST_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/ColoredMNIST/ColoredMNIST_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type erm --learned_vis --emb_layer 0 --vis_layer 0


To run tensorboard: tensorboard --logdir=runs --bind_all
To view locally: http://login2.ls6.tacc.utexas.edu:6006/#projector


droclassifiers for train + val, and drodataset for test doesn't work
analysis.py doens't work

for find groups:
python run_expt.py -s confounder -d ColoredMNIST -t target -c confounder --batch_size 32 --root_dir ./ --n_epochs 5 --aug_col None --log_dir results/ColoredMNIST/ColoredMNIST_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/ColoredMNIST/ColoredMNIST_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type erm --emb_layer 0 --emb_to_groups


python run_expt.py -s confounder -d ColoredMNIST_HARD -t target -c confounder --batch_size 32 --root_dir ./ --n_epochs 5 --aug_col None --log_dir results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type erm --emb_layer 0 --emb_to_groups


python run_expt.py -s confounder -d ColoredMNIST -t target -c confounder --batch_size 32 --root_dir ./ --n_epochs 5 --aug_col wrong_1_times --log_dir results/ColoredMNIST/ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/JTT_upweight_50_epochs_5_lr_0.001_weight_decay_0.0001_REWEIGHT/model_outputs --metadata_path results/ColoredMNIST/ColoredMNIST_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 50 --metadata_csv_name metadata.csv --model cnn --use_bert_params 1 --loss_type erm --upweight_misclassified --emb_layer 0 --emb_to_groups



TODO:
move redundant methods to confounder dataset
store predicted label
FIX get_embeddings to store misclassified or not in LR_targets

Need to fix group_str
Need to fix classifier groups

WANT: [0, 3, 0, 3, 3, 3, 2, 0, 2, 3, 3, 0, 0, 2, 3, 2, 1, 2, 3, 1, 0, 1, 2, 0,
        1, 2, 0, 2, 2, 0, 2, 0]

HAVE: [2, 0, 2, 1, 2, 3, 2, 2, 2, 1, 1, 3, 1, 0, 2, 1, 2, 1, 3, 1, 2, 1, 2, 1,
        1, 2, 2, 1, 2, 1, 2, 1]

HAVE even when I borrow default group array: [2, 0, 2, 1, 2, 3, 2, 2, 2, 1, 1, 3, 1, 0, 2, 1, 2, 1, 3, 1, 2, 1, 2, 1,
        1, 2, 2, 1, 2, 1, 2, 1]

WHY?

Bug candidate files:
folds.py - eliminated


TODO:
push changes
finish the new get_groups function
try on cmnist_hard