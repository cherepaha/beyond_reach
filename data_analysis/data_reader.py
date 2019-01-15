import pandas as pd
import numpy as np
import os.path
import derivative_calculator


class DataReader():
    index = ['subj_id', 'task', 'trial_no']



    def detect_backward_movement(self, choices, dynamics):
        choices['first_backwards'] = dynamics.groupby(by=self.index).apply(self.get_first_neg_vy)
        return choices

    def detect_first_slowdown(self, choices, dynamics):

        dynamics.reset_index(drop=False, inplace=True)
        dynamics.set_index(['subj_id', 'trial_no'], inplace=True, drop=True)
        choices['slowdown'] = dynamics.groupby(by=self.index).apply(self.get_slowdown)

        return choices


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
