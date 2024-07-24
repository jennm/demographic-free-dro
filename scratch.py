import pandas as pd

# df = pd.read_csv('/scratch/09717/saloni/demographic-free-dro/results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/metadata_aug.csv')
# column_sum = df['wrong_1_times'].sum()
# print(f"The sum of the column is: {column_sum}")

df = pd.read_csv('/scratch/09717/saloni/demographic-free-dro/results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/model_outputs/output_train_epoch_4.csv')

print('Predicted positive', df["y_pred_None_epoch_4_val"].sum())
print('Real positive', df["y_true_None_epoch_4_val"].sum())

print((df["y_pred_None_epoch_4_val"] != df["y_true_None_epoch_4_val"]).sum())

