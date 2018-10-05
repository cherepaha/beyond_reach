import csv
import numpy as np
import os
from collections import namedtuple
import constants

class WalkingExpDA:  
    choice_info_fields = ['subj_id', 'trial_no', 'task', 'order', 'is_staircase', 
                          'ss_delay', 'ss_amount', 'll_delay', 'll_amount', 'is_ss_on_left', 
                          'response', 'option_chosen', 'trial_time', 'start_time']
    ChoiceInfo = namedtuple('ChoiceInfo', choice_info_fields)
    
    mouse_response_log_fields = ['subj_id', 'trial_no', 'timestamp', 'x', 'y']
    walking_response_log_fields = np.concatenate([mouse_response_log_fields, 
                        np.array(list([[joint.name + '_x', joint.name + '_h', joint.name + '_y'] 
                            for joint in constants.TRACKED_JOINTS])).flatten()])
    
    def __init__(self, exp_info):
        self.initialize_log(exp_info)

    def initialize_log(self, exp_info):
        log_path = 'C:/Users/Arkady/Google Drive/data/beyond_the_reach/%s'
        
        if not os.path.exists(log_path % 'choices'):
            os.makedirs(log_path % 'choices')
        
        if not os.path.exists(log_path % 'dynamics'):
            os.makedirs(log_path % 'dynamics')
        
        log_name = log_path + '/' + str(exp_info['subj_id']) + '_' + exp_info['task'] + \
                '_' + exp_info['start_time'] + '_%s.txt'

        self.choices_log_file = log_name % ('choices', 'choices')
        self.response_dynamics_log_file = log_name % ('dynamics', 'dynamics')

        with open(self.choices_log_file, 'a') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerow(self.choice_info_fields)
                
        with open(self.response_dynamics_log_file, 'a') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            if (exp_info['task'] == 'mouse'):
                writer.writerow(self.mouse_response_log_fields)
            if (exp_info['task'] == 'walking'):
                writer.writerow(self.walking_response_log_fields)

    def write_trial_log(self, choice_info, response_dynamics_log, walking_log=None):
        with open(self.choices_log_file, 'a') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerow(choice_info._asdict().values())
            
        with open(self.response_dynamics_log_file, 'a') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerows(response_dynamics_log)

    def get_trial_params(self, filename):
        currency_rate = 100 #% scaling the USD amount
        trial_params = np.loadtxt(filename, skiprows = 1, delimiter = ',')
        trial_params[:,[1,3]]=currency_rate*trial_params[:,[1,3]]
        # zeros indicate mcq trials
        trial_params = np.column_stack((trial_params, np.zeros(len(trial_params)))).astype(int)
        return trial_params
        