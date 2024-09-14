import pandas as pd
import torch

seed = 42
torch.manual_seed(seed)
# df = pd.read_csv('/scratch/09717/saloni/demographic-free-dro/results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/metadata_aug.csv')
# column_sum = df['wrong_1_times'].sum()
# print(f"The sum of the column is: {column_sum}")

epoch = 4
# df = pd.read_csv(f'/scratch/09717/saloni/demographic-free-dro/coloredMNIST_HARD/data/metadata.csv')
df = pd.read_csv('/scratch/09717/saloni/demographic-free-dro/celebA/data/metadata.csv')

# print(df[(df['split'] == 0) & (df['target'] == 1) & (df['C4'] == 1)].shape[0])

# print("Split Size")
# print(df[df['split'] == 0].shape[0])
# print(df[df['split'] == 1].shape[0])
# print(df[df['split'] == 2].shape[0])

print("Train Target = 0 Counts by Color")
print(df[(df['split'] == 0) & (df['Male'] == -1) & (df['Blond_Hair'] == 1)].shape[0])
print(df[(df['split'] == 0) & (df['Male'] == -1) & (df['Blond_Hair'] == -1)].shape[0])
# print(df[(df['split'] == 0) & (df['target'] == 0) & (df['C2'] == 1)].shape[0])
# print(df[(df['split'] == 0) & (df['target'] == 0) & (df['C3'] == 1)].shape[0])
# print(df[(df['split'] == 0) & (df['target'] == 0) & (df['C4'] == 1)].shape[0])

print("Train Target = 1 Counts by Color")
print(df[(df['split'] == 0) & (df['Male'] == 1) & (df['Blond_Hair'] == 1)].shape[0])
print(df[(df['split'] == 0) & (df['Male'] == 1) & (df['Blond_Hair'] == -1)].shape[0])
# print(df[(df['split'] == 0) & (df['target'] == 1) & (df['C2'] == 1)].shape[0])
# print(df[(df['split'] == 0) & (df['target'] == 1) & (df['C3'] == 1)].shape[0])
# print(df[(df['split'] == 0) & (df['target'] == 1) & (df['C4'] == 1)].shape[0])

# print("Val Target Counts by Color")
# print(df[(df['split'] == 1) & (df['target'] == 1) & (df['C0'] == 1)].shape[0])
# print(df[(df['split'] == 1) & (df['target'] == 1) & (df['C1'] == 1)].shape[0])
# print(df[(df['split'] == 1) & (df['target'] == 1) & (df['C2'] == 1)].shape[0])
# print(df[(df['split'] == 1) & (df['target'] == 1) & (df['confounder'] == 1)].shape[0])
# print(df[(df['split'] == 1) & (df['target'] == 1) & (df['C4'] == 1)].shape[0])

# print("Test Target Counts by Color")
# print(df[(df['split'] == 2) & (df['target'] == 1) & (df['C0'] == 1)].shape[0])
# print(df[(df['split'] == 2) & (df['target'] == 1) & (df['C1'] == 1)].shape[0])
# print(df[(df['split'] == 2) & (df['target'] == 1) & (df['C2'] == 1)].shape[0])
# print(df[(df['split'] == 2) & (df['target'] == 1) & (df['confounder'] == 1)].shape[0])
# print(df[(df['split'] == 2) & (df['target'] == 1) & (df['C4'] == 1)].shape[0])



