import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class DataAnalyser():
    def get_indifference_point_staircase(self, choices, delay):
        if len(choices[~choices.ss_chosen])==0:
            ip = 0
        elif len(choices[choices.ss_chosen])==0:
             ip = 1
        else:
            ip = (choices[choices.ss_chosen].amount_ratio.min()
                    + choices[~choices.ss_chosen].amount_ratio.max())/2
        return ip
    
    def get_indifference_points(self, choices_sc):
        indiff_points = (choices_sc.groupby(['subj_id', 'task', 'll_delay'])
                         .apply(lambda c: self.get_indifference_point_staircase(c, c.iloc[0].ll_delay))
                         .rename('indiff_point'))
        return indiff_points
    
    def get_k(self, indiff_points, log_delay=False):
        '''
        Many "raw" k-values are close to 1, and to emphasize the differences between those, we
        might (or might not?) want to log-scale the delays before calculating k-values
        '''
        delays = indiff_points.ll_delay.unique()
        if log_delay:
            delays = np.log(delays)
        
        delays = delays/max(delays)        
        values = indiff_points.indiff_point.values
        
        delays = np.append(0, delays)
        values = np.append(1, values)
        
        k = 1 - ((delays[1:] - delays[:-1]) * (values[:-1] + values[1:]) / 2).sum()
    
        return k
    
    def get_k_values(self, choices, log=False):
        ip = self.get_indifference_points(choices[choices.is_staircase]).reset_index()
    
        k_values = ip.groupby(['subj_id', 'task']).apply(lambda x: self.get_k(x, log_delay=False)).rename('k-value').unstack().reset_index()
        k_values_log = ip.groupby(['subj_id', 'task']).apply(lambda x: self.get_k(x, log_delay=True)).rename('k-value').unstack().reset_index()
        if log:
            k_values = k_values_log
           
        return k_values, ip
    
    def get_long_k_values(self, k_values, choices):
        k_values_long = pd.melt(k_values, id_vars=['subj_id'], value_vars=['mouse', 'walking'], value_name='k')
        k_values_long = k_values_long.join(choices.groupby('subj_id').order.first(), on='subj_id')
        k_values_long['task_order'] = 1
        k_values_long.loc[(((k_values_long.task=='mouse') & (k_values_long.order=='wm')) | 
                      ((k_values_long.task=='walking') & (k_values_long.order=='mw'))), ['task_order']] = 2
        k_values_long = k_values_long.drop(['order'], axis=1)
        k_values_long = k_values_long.rename(columns={'task_order':'order'})
        
        return k_values_long
    
    def get_ss_bias(self, data_path):
        coeffs = pd.read_csv(os.path.join(data_path, 'mouse_max_d_coeffs.csv'), sep=',')
        coeffs.columns = ['name', 'SS_bias']
        coeffs = coeffs[coeffs['name'].str.contains(',choice')]
        coeffs['subj_id'] = coeffs.name.str.extract('(\d+)').astype(np.int64)
        coeffs = coeffs.drop('name', axis=1)
        
        return coeffs
    
    def plot_trajectories(self, trajectories, kind='xy', color='green', ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
    
        for name, traj in trajectories.groupby(level='trial_no'):
            if kind=='xy':
                ax.plot(traj.x, traj.y, color=color, alpha=0.5)
            elif kind=='time':
                ax.plot(traj.t, traj.x, color=color, alpha=0.5)
    
    
    def plot_trajectories_by_subject(self, dynamics, subjects):
        f, axes = plt.subplots(ncols=min(5, len(subjects)), sharex=True, sharey=True, squeeze=True, figsize=(13, 4))
        for i, subj_id in enumerate(subjects):
            self.plot_trajectories(dynamics[dynamics.option_chosen == 'ss'].loc[subj_id], color='C0', ax=axes[i])
            self.plot_trajectories(dynamics[dynamics.option_chosen == 'll'].loc[subj_id], color='C1', ax=axes[i])
            axes[i].set_title(subj_id)
    
        f.tight_layout()
        f.show()                                    
