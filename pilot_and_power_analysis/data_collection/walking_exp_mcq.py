from walking_exp_ui import WalkingExpUI
from walking_exp_da import WalkingExpDA
from datetime import datetime
import time

class WalkingExp:   
    def __init__(self, user_interface, data_access):
        self.user_interface = user_interface
        self.data_access = data_access

    def run_exp(self, n = None):
        trial_params = self.data_access.get_trial_params()
        if n is None:
            n = len(trial_params)
            
        for i in range(1, n+1):
            trial_info = self.run_trial(i, trial_params[i-1])
            self.data_access.write_trial_log(trial_info)
    
    def run_trial(self, trial_no, params):
        self.user_interface.show_messages('Ready?')
        self.user_interface.show_messages('Go!')
#        self.user_interface.show_messages(' ')
        trial_start_time_str = datetime.strftime(datetime.now(), '%Y-%m-%d--%H-%M-%S')
        trial_start_time = time.clock()
        response = self.user_interface.show_choices(params)
        response_time = time.clock() - trial_start_time            

        return [self.data_access.exp_info['subj_id'], trial_no,
                params[0], params[1], params[2], params[3],
                self.user_interface.is_ss_left, response, 
                response_time, trial_start_time_str]
                
user_interface = WalkingExpUI(n_screens = 1)   
data_access = WalkingExpDA()   
we = WalkingExp(user_interface, data_access)
#user_interface.show_messages('Left screen', 'Right screen')
we.run_exp(4)
user_interface.show_messages('Game over!')
