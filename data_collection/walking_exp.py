import ui, da
from datetime import datetime
import time, math
import numpy as np
import Tkinter as tkr

class WalkingExp:   
    sc_steps = [400, 200, 100, 0]
    sc_delays = [7, 30, 183, 365, 1095]

    def get_exp_info(self):
        root = tkr.Tk()
        app = ui.WalkingExpInfoUI(master=root)
        app.mainloop()
        return app.exp_info
    
    def __init__(self):        
        self.exp_info = self.get_exp_info()
        print(self.exp_info)
        
        self.data_access = da.WalkingExpDA(self.exp_info)
        
        if self.exp_info['task'] == 'mouse':
            self.user_interface = ui.WalkingExpUIMouse()
        elif self.exp_info['task'] == 'walking':
            self.user_interface = ui.WalkingExpUIWalking()
        
        self.mcq_trial_params = self.data_access.get_trial_params('mcq.csv')
    
    def run_practice(self, task='mouse', n_trials=12):
        practice_trial_params = self.data_access.get_trial_params('practice.csv')
        self.user_interface.show_messages('Left screen', 'Right screen')
        
        for i, trial in enumerate(practice_trial_params):
            self.run_trial(i, trial)
    
    def run_exp(self, n_blocks=4):
        self.user_interface.show_messages('Left screen', 'Right screen')
        
        # sc_state contains the current ss_amount for each of ll_delays
        sc_state = { delay : 800 for delay in self.sc_delays } 
        for i in range(1, n_blocks+1):
            sc_state = self.run_block(i, sc_state)
        self.user_interface.show_messages('End of experiment')
    
    def run_block(self, block_no, sc_state):
        mcq_trial_no = math.ceil(len(self.mcq_trial_params)/4.0)
        mcq_trials = self.mcq_trial_params[int((block_no-1)*mcq_trial_no):\
                                    int(min(block_no*mcq_trial_no, len(self.mcq_trial_params)))]      
        # the order is: 'ss_delay', 'ss_amount', 'll_delay', 'll_amount', 'is_staircase'
        sc_trials = [[0, sc_state[delay], delay, 1600, True] for delay in self.sc_delays]
        
        trials = np.concatenate((mcq_trials, sc_trials))
        np.random.shuffle(trials)

        for i, trial_params in enumerate(trials):
            choice_info, response_dynamics_log = self.run_trial((block_no-1)*12+i+1, trial_params)
            # if staircase trial, update the rewards
            if trial_params[4]==1:
                # if ss was chosen, decrease ss reward
                if choice_info.option_chosen == 'ss':
                    sc_state[trial_params[2]] -= self.sc_steps[block_no-1]
                # if ll was chosen, increase ss reward
                else:
                    sc_state[trial_params[2]] += self.sc_steps[block_no-1]
            self.data_access.write_trial_log(choice_info, response_dynamics_log)     
        return sc_state

    def run_trial(self, trial_no, params):
        trial_info = {'task': self.exp_info['task'],
                      'order': self.exp_info['order'],
                      'subj_id': self.exp_info['subj_id'],
                      'trial_no': trial_no}
        
        self.user_interface.show_trial_start_screen()
        
        trial_start_time_str = datetime.strftime(datetime.now(), '%Y-%m-%d--%H-%M-%S')
        trial_start_time = time.clock()
        
        response, response_dynamics_log = self.user_interface.show_choices(params, trial_info)
        trial_time = time.clock() - trial_start_time        
        option_chosen = 'ss' if (((self.user_interface.is_ss_left) & (response=='left')) or 
                                 ((not self.user_interface.is_ss_left) & (response=='right'))) else 'll'
       
        choice_info = da.WalkingExpDA.ChoiceInfo(
                subj_id=trial_info['subj_id'],
                trial_no=trial_no,
                task=trial_info['task'], 
                order=trial_info['order'],
                is_staircase=params[4],
                ss_delay=params[0], 
                ss_amount=params[1], 
                ll_delay=params[2], 
                ll_amount=params[3],
                is_ss_on_left=self.user_interface.is_ss_left, 
                response=response,
                option_chosen=option_chosen, 
                trial_time=trial_time, 
                start_time=trial_start_time_str)
        print(choice_info)

        return choice_info, response_dynamics_log