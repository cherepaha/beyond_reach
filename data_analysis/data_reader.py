import pandas as pd
import numpy as np
import os.path
import derivative_calculator


class DataReader():
    index = ['subj_id', 'task', 'trial_no']

    def read_data(self, path='data'):
        '''
        When reading the data, we set the index to self.index to be able to transfer the results of 
        processing the dynamics dataframe to the choices dataframe. This is also convenient for data 
        wrangling (e.g., dynamics.loc[subj_id, task, trial_no]). However, we also keep index as 
        columns (drop=False) for more convenient pre-processing and plotting. Having the same data 
        as a column and as an index is not very pythonic, but is justified here IMO. Because of this, 
        pandas throws tons of warnings here and there, but those can be ignored so far.        
        '''

        file_path = os.path.join(path, '%s.txt')

        choices = pd.read_csv(file_path % 'choices', sep='\t')
        choices.set_index(self.index, inplace=True, drop=False)
        dynamics = pd.read_csv(file_path % 'dynamics', sep='\t')
        dynamics.set_index(self.index, inplace=True, drop=False)

        dynamics = dynamics.rename(columns={'timestamp': 't'})

        return choices, dynamics

    def detect_backward_movement(self, choices, dynamics):
        choices['first_backwards'] = dynamics.groupby(by=self.index).apply(self.get_first_neg_vy)
        return choices

    def detect_first_slowdown(self, choices, dynamics):

        dynamics.reset_index(drop=False, inplace=True)
        dynamics.set_index(['subj_id', 'trial_no'], inplace=True, drop=True)
        choices['slowdown'] = dynamics.groupby(by=self.index).apply(self.get_slowdown)

        return choices

    def shift_timeframe(self, dynamics):
        # shift time to the timeframe beginning at 0 for each trajectory
        # also, express time in seconds rather than milliseconds
        dynamics.loc[:, 'timestamp'] = dynamics.timestamp.groupby(by=self.index). \
                                           transform(lambda t: (t - t.min())) / 1000.0
        return dynamics

    def apply_trajectory_resampling(self, dynamics):
        dynamics = dynamics.groupby(by=self.index).apply(self.resample_trajectory)
        return dynamics

    def add_basic_measures(self, choices):

        # This method adds measures that aren't necessarily *affected* by any pre-processing steps.
        choices['is_staircase'] = choices['is_staircase'].astype('bool')

        choices['ss_chosen'] = ((choices['is_ss_on_left']) == (choices.response == 'left'))
        choices['choice'] = 'SS'
        choices.loc[~choices.ss_chosen, 'choice'] = 'LL'

        choices['chosen_amount'] = (choices['ss_amount'] * choices['ss_chosen'] +
                                    choices['ll_amount'] * (~choices['ss_chosen']))
        choices['chosen_delay'] = (choices['ss_delay'] * choices['ss_chosen'] +
                                   choices['ll_delay'] * (~choices['ss_chosen']))
        choices['amount_ratio'] = choices['ss_amount'] / choices['ll_amount']

        choices['amount_diff'] = choices['ll_amount'] - choices['ss_amount']
        choices['amount_increase'] = (choices['ll_amount'] - choices['ss_amount']) / choices['ss_amount']
        choices['LL_advantage'] = choices['amount_diff'] / choices['ll_delay']
        choices['Lambda'] = np.log(choices['LL_advantage'])
        # choices['type'] = 'MCQ'
        # choices.loc[choices.is_staircase, 'type'] = 'Staircase'

        choices.drop(['is_ss_on_left', 'response', 'start_time'], axis=1, inplace=True)

        return choices

    def get_RT(self, choices, dynamics):

        choices = choices.join(dynamics.groupby(by=self.index).apply(self.get_maxd), on=self.index)
        choices['RT'] = dynamics.groupby(by=self.index).apply(lambda traj: traj.t.max() - traj.t.min())

        # TODO: z-scoring should work within task (walking/mouse)
        #        choices['max_d_z'] = (choices['max_d'] - choices['max_d'].mean()) / choices['max_d'].std()

        #        choices.reset_index(drop=False, inplace=True)
        #        choices.set_index(self.index, inplace=True, drop=True)

        return choices

    def get_maxd(self, trajectory):
        alpha = np.arctan((trajectory.y.iloc[-1] - trajectory.y.iloc[0]) / \
                          (trajectory.x.iloc[-1] - trajectory.x.iloc[0]))
        d = (trajectory.x.values - trajectory.x.values[0]) * np.sin(-alpha) + \
            (trajectory.y.values - trajectory.y.values[0]) * np.cos(-alpha)

        if np.isnan(d).all():
            return pd.Series({'max_d': np.nan, 'idx_max_d': np.nan})
        elif abs(np.nanmin(d)) > abs(np.nanmax(d)):
            return pd.Series({'max_d': np.nanmin(d), 'idx_max_d': np.nanargmin(d)})
        else:
            return pd.Series({'max_d': np.nanmax(d), 'idx_max_d': np.nanargmax(d)})

    def resample_trajectory(self, trajectory, derivatives=False):
        # Make the sampling time intervals regular, this is only needed for average trajectories/decision landscapes
        n_steps = 50
        t_regular = np.linspace(trajectory.t.min(), trajectory.t.max(), n_steps + 1)
        x_interp = np.interp(t_regular, trajectory.t.values, trajectory.x.values)
        y_interp = np.interp(t_regular, trajectory.t.values, trajectory.y.values)
        vx_interp = np.interp(t_regular, trajectory.t.values, trajectory.vx.values)
        vy_interp = np.interp(t_regular, trajectory.t.values, trajectory.vy.values)
        traj_interp = pd.DataFrame([t_regular, x_interp, y_interp,
                                    vx_interp, vy_interp]).transpose()
        traj_interp.columns = ['t', 'x', 'y', 'vx', 'vy']

        return traj_interp

    # TODO: update preprocessing

    # TODO: 1- Remove backward movement from the kinect. 2- Remove cut trajectories that go outside the screen limit?

    def get_first_neg_vy(self, traj):
        neg_vy = np.argwhere(traj.vy < 0)
        return np.nan if len(neg_vy) == 0 else neg_vy[0][0]

    # def get_slowdown(self, traj):
    #     slowdown = np.argwhere(traj.vy < 0.2)
    #
    #     if(np.median(slowdown)< len(traj) - 30):
    #         slow = 1
    #     else:
    #         slow = np.nan
    #
    #     return slow

    def get_outside_screen_limit(traj):
        return

    def rotate_trajectory(self, trajectory, alpha):
        trajectory.x = trajectory.x * np.cos(alpha) - trajectory.y * np.sin(alpha)
        trajectory.y = trajectory.x * np.sin(alpha) + trajectory.y * np.cos(alpha)
        return trajectory

    def transform_xy(self, dynamics):
        # first, extract average of the left and right shoulder markers data
        # also, we rotate the reference frame so that x is "horizontal" motion (between the options)
        # and y is "vertical" (towards/away from the options)
        # dynamics['x'] = -(dynamics['y_left'] + dynamics['y_right']) / 2.0
        # dynamics['y'] = (dynamics['x_left'] + dynamics['x_right']) / 2.0
        # dynamics.drop(['x_left', 'y_left', 'z_left', 'x_right', 'y_right', 'z_right'],
        #               axis=1, inplace=True)
        #
        # # second, correct camera calibration error: rotate every subject's trajectories
        # # so that the locations of the response areas are symmetric
        # end_points = dynamics.groupby(by=self.index).last()
        # mean_end_points_pos = end_points[end_points.x > 0].groupby('subj_id').mean()
        # mean_end_points_neg = end_points[end_points.x < 0].groupby('subj_id').mean()
        # dynamics = dynamics.reset_index(drop=True)
        # for subj_id in dynamics.subj_id.unique():
        #     alpha = np.arctan((mean_end_points_neg.loc[subj_id].y - mean_end_points_pos.loc[subj_id].y) / \
        #                       (mean_end_points_pos.loc[subj_id].x - mean_end_points_neg.loc[subj_id].x))
        #     rotate = lambda traj: self.rotate_trajectory(traj, alpha)
        #     dynamics[dynamics.subj_id == subj_id] = \
        #         dynamics[dynamics.subj_id == subj_id].groupby('trial_no').apply(rotate)
        #
        # dynamics = dynamics.set_index(keys=self.index)

        # third, shift every trajectory so that on average they all start at (0,0)
        starting_points = dynamics.groupby(by=self.index).first()
        dynamics.x = dynamics.x - starting_points.mean(axis=0).x
        dynamics.y = dynamics.y - starting_points.mean(axis=0).y

        return dynamics

    def apply_trajectory_resampling(self, dynamics):
        dynamics = dynamics.groupby(by=self.index).apply(self.resample_trajectory)
        dynamics.reset_index(drop=False, inplace=True)
        del dynamics['level_3']
        dynamics.set_index(self.index, inplace=True, drop=True)
        return dynamics
