import numpy as np
import pandas as pd
from numpy import random

def generate_subj_ids(n):
    if n % 2 != 0:
        raise Exception('n should be even!')
    subj_ids = random.uniform(1000, 9999, n).astype(int)
    task_order = np.concatenate([np.repeat('mw', n/2), np.repeat('wm', n/2)])
    random.shuffle(task_order)
    result = pd.DataFrame({'subj_id': subj_ids, 'order': task_order})
    result.to_csv('remaining_subj_ids.csv', index=False)
    
generate_subj_ids(n=30)   
    