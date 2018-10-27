import numpy as np
import pandas as pd
from numpy import random
import data_reader

def generate_subj_ids(n, existing_ids):
    if n % 2 != 0:
        raise Exception('n should be even!')
    
    subj_ids = random.uniform(1000, 9999, n).astype(int)
    task_order = np.concatenate([np.repeat('mw', n/2), np.repeat('wm', n/2)])
    random.shuffle(task_order)
    df = pd.DataFrame({'subj_id': subj_ids, 'order': task_order})
    
    while len(list(set(subj_ids) & set(existing_ids)))>0:
        subj_ids, df = generate_subj_ids(n, existing_ids)
        
    return df
    
dr = data_reader.DataReader()
choices, dynamics = dr.read_data('C:/Users/Arkady/Google Drive/data/beyond_the_reach')
existing_ids = choices.index.get_level_values(0).unique()
result = generate_subj_ids(n=10, existing_ids=existing_ids)
result.to_csv('remaining_subj_ids.csv', index=False)