import data_reader, data_analyser
import os

data_path = 'C:/Users/Arkady/Google Drive/data/beyond_the_reach'
da = data_analyser.DataAnalyser()
dr = data_reader.DataReader()
choices, dynamics = dr.read_data(data_path)
choices, dynamics = dr.preprocess_data(choices, dynamics)

print(len(choices)/47)

k_values, ip = da.get_k_values(choices, log=False)

# excluding participants based on extreme discounting behavior
extreme_k = k_values.loc[((k_values.mouse>0.98)&(k_values.walking>0.98)) | 
                         ((k_values.mouse<0.02)&(k_values.walking<0.02)), 'subj_id']

print(len(extreme_k))
print(extreme_k.values)

choices = choices[~choices.subj_id.isin(extreme_k)]
dynamics = dynamics[~dynamics.subj_id.isin(extreme_k)]

# exclude trials with more than x% data loss?

# we have about 2.5% data loss overall in the walking data
len(dynamics[(dynamics.task=='walking')&(dynamics.x.isna())])/len(dynamics[dynamics.task=='walking'])

def get_data_loss_rate(trajectory):
    return len(trajectory[(trajectory.x.isna())])/len(trajectory)

data_loss_rate = dynamics.groupby(['subj_id', 'task', 'trial_no']).apply(get_data_loss_rate)

# how many trials are there with more than, say, 20% data loss?
print(data_loss_rate[data_loss_rate>0.2])
# just 7

# TODO: drop these trials

# exclude trials with extreme response times


# exclude subjects with less than 80% trials left after all exclusions (47*0.8 ~~ 38)


# saving preprocessed choices and dynamics for Analysis 1
choices.to_csv(os.path.join(data_path, 'choices_processed.txt'), index=False)
dynamics.to_csv(os.path.join(data_path, 'dynamics_processed.txt'), index=False)


# saving k-values for Analysis 2
k_values, ip = da.get_k_values(choices, log=False)
k_values_long = da.get_long_k_values(k_values, choices)

bias = da.get_ss_bias(data_path)
k_values_long = k_values_long.join(bias.set_index('subj_id'), on='subj_id')

k_values_long.to_csv(os.path.join(data_path, 'k_values.csv'), sep='\t', index=False)

