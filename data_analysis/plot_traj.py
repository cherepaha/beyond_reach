import matplotlib.pyplot as plt
import data_reader

def plot_subject_trajectories(trajectories, color='green', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    
    for name, traj in trajectories.groupby(level='trial_no'):
        ax.plot(traj.x, traj.y, color=color, alpha=0.5)                                

def plot_trajectories(dynamics, subjects):
    f, axes = plt.subplots(ncols=5, sharex=True, sharey=True, squeeze=True, figsize=(13,4))
    
    for i, subj_id in enumerate(subjects):
        plot_subject_trajectories(dynamics[dynamics.option_chosen=='ss'].loc[subj_id], color='C0', ax=axes[i])
        plot_subject_trajectories(dynamics[dynamics.option_chosen=='ll'].loc[subj_id], color='C1', ax=axes[i])
        axes[i].set_title(subj_id)

dr = data_reader.DataReader()
choices, dynamics = dr.read_data('C:/Users/Arkady/Google Drive/data/beyond_the_reach')

dynamics = dynamics.join(choices.option_chosen)

subjects = choices.subj_id.unique()
    
plot_trajectories(dynamics[dynamics.task == 'walking'], subjects)
