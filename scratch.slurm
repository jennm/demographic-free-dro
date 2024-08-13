#!/bin/bash

#SBATCH -A CCR23048
#SBATCH -J GEORGE           # Job name
#SBATCH -o status/CMNIST_HARD_TEST.o%j       # Name of stdout output file
#SBATCH -e status/CMNIST_HARD_TEST.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-small         # Queue (partition) name
#SBATCH -N 1                # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00         # Run time (hh:mm:ss)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=saloni.a.modi@utexas.edu

set -e

cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/DRO2

cd demographic-free-dro

# python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --batch_size 32 --root_dir ./ --n_epochs 20 --aug_col None --log_dir results/CelebA/CelebA_SMALL_TEST/ERM_upweight_0_epochs_50_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/CelebA/CelebA_SMALL_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type erm 
# python run_expt.py -s confounder -d ColoredMNIST_HARD -t target -c confounder --batch_size 32 --root_dir ./ --n_epochs 19 --aug_col None --log_dir results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type erm # --emb_to_groups --emb_layers 0 1 --resume_path 4_model.pth
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --batch_size 32 --root_dir ./ --n_epochs 20 --aug_col None --log_dir results/CelebA/CelebA_SMALL_TEST/ERM_upweight_0_epochs_50_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/CelebA/CelebA_SMALL_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type erm --emb_to_groups --emb_layers 0 1 --resume_path 12_model.pth
# python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --batch_size 32 --root_dir ./ --n_epochs 20 --aug_col None --log_dir results/CelebA/CelebA_SMALL_TEST/GROUP_DRO_upweight_0_epochs_50_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/CelebA/CelebA_SMALL_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type group_dro --reweight_groups

# python run_expt.py -s confounder -d ColoredMNIST_HARD -t target -c confounder --batch_size 32 --root_dir ./ --n_epochs 20 --aug_col None --log_dir results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/model_outputs --metadata_path results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/metadata_aug.csv --lr 0.001 --weight_decay 0.0001 --up_weight 0 --metadata_csv_name metadata.csv --model cnn --use_bert_params 0 --loss_type erm # --emb_to_groups --emb_layers 0 1 --resume_path 14_model.pth

