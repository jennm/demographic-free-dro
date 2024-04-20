import pandas as pd

def get_info(df):
    target_0_confounder_0 = ((df['target'] == 0) & (df['confounder'] == 0)).sum()
    target_0_confounder_1 = ((df['target'] == 0) & (df['confounder'] == 1)).sum()
    target_1_confounder_0 = ((df['target'] == 1) & (df['confounder'] == 0)).sum()
    target_1_confounder_1 = ((df['target'] == 1) & (df['confounder'] == 1)).sum()

    return target_0_confounder_0, target_0_confounder_1, target_1_confounder_0, target_1_confounder_1

def get_metrics(file_path):
    df = pd.read_csv(file_path)
    # target_confounder
    df_split = df[df['split'] == 0]
    df_val = df[df['split'] == 1]
    total_train = len(df_split)
    total_val = len(df_val)
    
    train_info = get_info(df_split)
    val_info = get_info(df_val)
    print("Target Confounder")
    target_confounder = ['0 0:', '0 1:', '1 0:', '1 1:']
    train_string = ''
    for i in range(len(target_confounder)):
        train_string += f'{target_confounder[i]} {train_info[i]} ({train_info[i] / total_train * 100}%) '
    
    val_string = ''
    for i in range(len(target_confounder)):
        val_string += f'{target_confounder[i]} {val_info[i]} ({val_info[i] / total_val * 100}%) '
    
    print(train_string)
    print(val_string)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="filepath to dataset metadata.csv file"
    )

    args = parser.parse_args()

    get_metrics(args.dataset_path)