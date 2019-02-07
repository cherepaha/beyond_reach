import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class HelperFunctions():
    index = ['subj_id', 'task', 'trial_no']
    
    def append_derivatives(self, dynamics):
        names = {'x': 'vx',
                 'y': 'vy'}

        for col_name, der_name in names.items():
            dynamics[der_name] = np.concatenate(
                    [self.differentiate(traj['t'].values, traj[col_name].values)
                            for traj_id, traj in dynamics.groupby(by=self.index, group_keys=False)]
                    )

        return dynamics
    
    def get_diff(self, t, x):
        return np.concatenate(([0.], np.diff(x)/np.diff(t)))
    
    def differentiate(self, t, x):
        # To be able to reasonably calculate derivatives at the end-points of the trajectories,
        # I append three extra points before and after the actual trajectory, so we get N+6
        # points instead of N       
        x = np.append(x[0]*np.ones(3), np.append(x, x[-1]*np.ones(3)))
        
        # Time vector is also artificially extended by equally spaced points
        # Use median timestep to add dummy points to the time vector
        timestep = np.median(np.diff(t))
        t = np.append(t[0]-np.arange(1,4)*timestep, np.append(t, t[-1]+np.arange(1,4)*timestep))

        # smooth noise-robust differentiators, see: 
        # http://www.holoborodko.com/pavel/numerical-methods/ \
        # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
        v = (1*(x[6:]-x[:-6])/((t[6:]-t[:-6])/6) + 
             4*(x[5:-1] - x[1:-5])/((t[5:-1]-t[1:-5])/4) + 
             5*(x[4:-2] - x[2:-4])/((t[4:-2]-t[2:-4])/2))/32
        
        return v
    
    def get_indifference_point_staircase(self, choices, delay):
        if len(choices[~choices.ss_chosen])==0:
            ip = 0
        elif len(choices[choices.ss_chosen])==0:
             ip = 1
        else:
            ip = (choices[choices.ss_chosen].amount_ratio.min()
                    + choices[~choices.ss_chosen].amount_ratio.max())/2
        return ip
    
    def get_indifference_points(self, choices_sc, by='task'):
        indiff_points = (choices_sc.groupby(['subj_id', by, 'll_delay'])
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
    
    def get_k_values(self, choices, by='task', log=False):
        ip = self.get_indifference_points(choices[choices.is_staircase], by=by).reset_index()
        
        k_values = (ip.groupby(['subj_id', by]).apply(lambda x: self.get_k(x, log_delay=log)).
                        rename('k-value').unstack().reset_index())
           
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
    
    def plot_trajectories(self, trajectories, kind='xy', color='C0', ax=None, aspect='auto'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect=aspect)
    
        for name, traj in trajectories.groupby(level=trajectories.index.names):
            if kind=='xy':                
                ax.plot(traj.x, traj.y, color=color, alpha=0.5)
            elif kind=='time':
                ax.plot(traj.t, traj.x, color=color, alpha=0.5)
    
    
    def plot_trajectories_by_subject(self, dynamics, subjects):
        ncols = min(5, len(subjects))
        nrows = int(np.ceil(len(subjects)/5))
        f, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, 
                               squeeze=True, figsize=(13, 3.5*nrows))
        for i, subj_id in enumerate(subjects):
            ax = axes.flatten()[i]
            self.plot_trajectories(dynamics[dynamics.option_chosen == 'ss'].loc[subj_id], 
                                   color='C0', ax=ax)
            self.plot_trajectories(dynamics[dynamics.option_chosen == 'll'].loc[subj_id], 
                                   color='C1', ax=ax)
            ax.set_title(subj_id)
        
        for ax in axes.flatten():
#            ax.set_aspect('equal', adjustable='datalim')
            ax.set_aspect('equal', adjustable='box')
#            ax.set_aspect('equal')
    
        f.tight_layout()
        f.show()                                    
