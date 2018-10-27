import os
import pandas as pd


def merge_choices(name, dir_path):
    subdir_path = os.path.join(dir_path, name)
    dfs = []
    for file in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file)
        if file_path.endswith('.txt'):
            print(file_path)
            dfs.append(pd.read_csv(file_path, sep='\t'))
    df_concat = pd.concat(dfs)
    df_concat.to_csv(os.path.join(dir_path, name + '.txt'), index=False, sep='\t')


def merge_dynamics(name, dir_path):
    subdir_path = os.path.join(dir_path, name)

    dfs = []
    for file in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file)
        if (file_path.endswith('.txt')):
            print(file_path)
            df = pd.read_csv(file_path, sep='\t').iloc[:, :5]
            df['task'] = 'walking' if 'walking' in file_path else 'mouse'
            dfs.append(df)
    df_concat = pd.concat(dfs)
    df_concat.to_csv(os.path.join(dir_path, name + '.txt'), index=False, sep='\t')


dir_path = '../data/'
#dir_path = 'C:/Users/Arkady/Google Drive/data/beyond_the_reach'

merge_choices('choices', dir_path)
merge_dynamics('dynamics', dir_path)
