import pandas as pd
import numpy as np
import os.path
from pydlv import derivative_calculator

class DataReader():
    index = ['subj_id', 'trial_no']
    def read_data(self, path='data'):
        drop_subj = 530
        drop_trials = [[967, 40], [967, 42], [967, 45]]
        # drop_trials = []
        file_path = os.path.join(path, '%s.txt')        
        
        choices = pd.read_csv(file_path % 'choices', sep='\t')
        # there was an error in the stimulus presentation code, incorrect headers were written to 
        # data files. this fixes the problem:
        choices.rename(columns={'left_delay': 'ss_delay', 
                                'left_amount': 'ss_amount',
                                'right_delay': 'll_delay', 
                                'right_amount': 'll_amount'}, inplace=True)
        choices = choices[choices.subj_id != drop_subj]
        for subj_trial in drop_trials:
            choices = choices[~((choices.subj_id==subj_trial[0])&(choices.trial_no==subj_trial[1]))]
        choices.set_index(self.index, inplace=True, drop=False)
        
        dynamics = pd.read_csv(file_path % 'dynamics', sep=',')
        dynamics = dynamics[dynamics.subj_id != drop_subj]    
        for subj_trial in drop_trials:
            dynamics = dynamics[~((dynamics.subj_id==subj_trial[0]) 
                                    & (dynamics.trial_no==subj_trial[1]))]
        dynamics.set_index(self.index, inplace=True, drop=False)
    
        return choices, dynamics
    
    def get_maxd(self, trajectory):
        alpha = np.arctan((trajectory.y.iloc[-1]-trajectory.y.iloc[0])/ \
                            (trajectory.x.iloc[-1]-trajectory.x.iloc[0]))
        d = (trajectory.x.values-trajectory.x.values[0])*np.sin(-alpha) + \
            (trajectory.y.values-trajectory.y.values[0])*np.cos(-alpha)
        if abs(d.min())>abs(d.max()):
            return pd.Series({'max_d': d.min(), 'idx_max_d': d.argmin()})
        else:
            return pd.Series({'max_d': d.max(), 'idx_max_d': d.argmax()})
    
    def rotate_trajectory(self, trajectory, alpha):
        trajectory.x = trajectory.x*np.cos(alpha) - trajectory.y*np.sin(alpha)
        trajectory.y = trajectory.x*np.sin(alpha) + trajectory.y*np.cos(alpha)
        return trajectory
        
    def transform_xy(self, dynamics):
        # first, extract average of the left and right shoulder markers data
        # also, we rotate the reference frame so that x is "horizontal" motion (between the options)
        # and y is "vertical" (towards/away from the options)
        dynamics['x'] = -(dynamics['y_left'] + dynamics['y_right'])/2.0
        dynamics['y'] = (dynamics['x_left'] + dynamics['x_right'])/2.0
        dynamics.drop(['x_left', 'y_left', 'z_left', 'x_right', 'y_right', 'z_right'], 
                      axis=1, inplace=True)
        
        # second, correct camera calibration error: rotate every subject's trajectories
        # so that the locations of the response areas are symmetric
        end_points = dynamics.groupby(by=self.index).last()
        mean_end_points_pos = end_points[end_points.x>0].groupby('subj_id').mean()
        mean_end_points_neg = end_points[end_points.x<0].groupby('subj_id').mean()
        dynamics = dynamics.reset_index(drop=True)
        for subj_id in dynamics.subj_id.unique():
            alpha = np.arctan((mean_end_points_neg.loc[subj_id].y-mean_end_points_pos.loc[subj_id].y)/ \
                            (mean_end_points_pos.loc[subj_id].x-mean_end_points_neg.loc[subj_id].x))
            rotate = lambda traj: self.rotate_trajectory(traj, alpha)
            dynamics[dynamics.subj_id==subj_id] = \
                dynamics[dynamics.subj_id==subj_id].groupby('trial_no').apply(rotate)
        
        dynamics = dynamics.set_index(keys=self.index)
    
        # third, shift every trajectory so that on average they all start at (0,0)
        starting_points = dynamics.groupby(by=self.index).first()
        dynamics.x = dynamics.x - starting_points.mean(axis = 0).x
        dynamics.y = dynamics.y - starting_points.mean(axis = 0).y
            
        return dynamics
        
    def resample_trajectory(self, trajectory, derivatives=False):
        # Make the sampling time intervals regular
        n_steps = 50
        t_regular = np.linspace(trajectory.t.min(), trajectory.t.max(), n_steps+1)
        x_interp = np.interp(t_regular, trajectory.t.values, trajectory.x.values)
        y_interp = np.interp(t_regular, trajectory.t.values, trajectory.y.values)
        vx_interp = np.interp(t_regular, trajectory.t.values, trajectory.vx.values)
        vy_interp = np.interp(t_regular, trajectory.t.values, trajectory.vy.values)
        traj_interp = pd.DataFrame([t_regular, x_interp, y_interp, 
                                    vx_interp, vy_interp]).transpose()
        traj_interp.columns = ['t', 'x', 'y', 'vx', 'vy']
    
        return traj_interp
    
    def preprocess_data(self, choices, dynamics):
        dynamics = self.transform_xy(dynamics)
                
        dc = derivative_calculator.DerivativeCalculator()
        dynamics = dc.append_derivatives(dynamics)
        # dropping backward movement (only observed in the end of the trial)
        ## this shows that there is no meaningful backward motion
        #def get_first_neg_vy(traj):
        #    neg_vy = np.argwhere(traj.vy<0)
        #    return np.nan if len(neg_vy)==0 else neg_vy[0][0]
        #choices['first_backwards'] = dynamics.groupby(by=self.index).apply(get_first_neg_vy)
        #choices['first_backwards'].hist(bins=20)
        dynamics = dynamics[dynamics.vy>=0]
        dynamics = dynamics.groupby(by=self.index).apply(self.resample_trajectory)
        dynamics.index = dynamics.index.droplevel(2)
        choices = choices.join(dynamics.groupby(by=self.index).apply(self.get_maxd))
        choices['RT'] = dynamics.groupby(by=self.index).apply(lambda traj: traj.t.max())
            
        choices['ss_chosen'] = ((choices['is_ss_on_left']) == (choices.response == 'left'))
        choices['choice'] = 'SS'
        choices.loc[~choices.ss_chosen, 'choice'] = 'LL'
        
        choices['chosen_amount'] = (choices['ss_amount']*choices['ss_chosen'] +
                            choices['ll_amount']*(~choices['ss_chosen']))
        choices['chosen_delay'] = (choices['ss_delay']*choices['ss_chosen'] +
                            choices['ll_delay']*(~choices['ss_chosen']))
        choices['amount_ratio'] = choices['ss_amount']/choices['ll_amount']

        choices['amount_diff'] = choices['ll_amount']-choices['ss_amount']
        choices['amount_increase'] = (choices['ll_amount']-choices['ss_amount'])/choices['ss_amount']
        choices['LL_advantage'] = choices['amount_diff']/choices['ll_delay']
        choices['type'] = 'MCQ'
        choices.loc[choices.is_staircase, 'type'] = 'Staircase'
        choices.drop(['is_ss_on_left', 'response', 'start_time'], axis=1, inplace=True)
        choices['max_d_z'] = (choices['max_d']-choices['max_d'].mean())/choices['max_d'].std()
        
        return choices, dynamics
    