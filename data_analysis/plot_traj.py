import matplotlib.pyplot as plt
import data_reader


def plot_subject_trajectories(trajectories, color='green', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    for name, traj in trajectories.groupby(level='trial_no'):
        ax.plot(traj.x, traj.y, color=color, alpha=0.5)


def plot_trajectories(dynamics, subjects, xlim, ylim):
    f, axes = plt.subplots(ncols=5, sharex=True, sharey=True, squeeze=True, figsize=(13, 4))
    for i, subj_id in enumerate(subjects):
        plot_subject_trajectories(dynamics[dynamics.option_chosen == 'ss'].loc[subj_id], color='C0', ax=axes[i])
        plot_subject_trajectories(dynamics[dynamics.option_chosen == 'll'].loc[subj_id], color='C1', ax=axes[i])
        axes[i].set_title(subj_id)
        axes[i].set_xlim(xlim)
        axes[i].set_ylim(ylim)

    f.tight_layout()
    f.show()


dr = data_reader.DataReader()
choices, dynamics = dr.read_data('../data/')

dynamics = dynamics.join(choices.option_chosen)

dynamics.reset_index(drop=False, inplace=True)

dynamics.set_index(['subj_id', 'task', 'trial_no'], inplace=True, drop=False)

subjects = choices.index.get_level_values('subj_id').unique().values

# attempt to remove out of limits trajectories before plotting mouse trajectories
# x_mask_1 = (dynamics['x'] >= -1.2)
# dynamics = dynamics.loc[x_mask_1]
# x_mask_2 = (dynamics['x'] <= 1.2)
# dynamics = dynamics.loc[x_mask_2]
# y_mask_1 = (dynamics['y'] >= 1)
#
# dynamics = dynamics.loc[y_mask_1]
# y_mask_2 = (dynamics['y'] <= 4.5)
# dynamics = dynamics.loc[y_mask_2]


plot_trajectories(dynamics[dynamics.task == 'mouse'], subjects, (-1.2, 1.2), (1, 4.5))

# attempt to remove backward movement before plotting walking trajectories
dynamics = dynamics[dynamics.vy >= 0]

plot_trajectories(dynamics[dynamics.task == 'walking'], subjects, (-1.2, 1.2), (1, 4.5))
