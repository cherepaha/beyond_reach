import csv, datetime, time
import numpy as np
from numpy import random
from psychopy import visual, event, core

class WalkingExp:
    # Parameters of stimulus presentation
    # in units of half screen size
    text_v_offset = 0.3
    text_height = 0.2
    
    trial_log_file = ''
    n_trial = 2
    subj_id = 0
    exp_info = {}
    
    # Parameters of Kinect setup
    # numeric parameters in meters?
    trajectory_path = 'monkey1b-pos1_2015-11-19.dat'
    start_threshold = 1.5    
    left_resp_area_center = [3.25, 1.45]
    right_resp_area_center = [3.25, -1.45]
    resp_area_radius = 0.25
    
    # set to True if automatic detection of trial start/end is needed
    auto_control = True
    
    def initialize_exp(self):
        self.left_win = visual.Window(units='norm', fullscr=True, screen=1) 
        self.right_win = visual.Window(units='norm', fullscr=True, screen=0)
        
        self.left_test = visual.TextStim(self.left_win, text = 'Left screen')
        self.right_test = visual.TextStim(self.right_win, text = 'Right screen')
        
        self.left_message = visual.TextStim(self.left_win, height = self.text_height)
        self.right_message = visual.TextStim(self.right_win, height = self.text_height)

        self.left_delay = visual.TextStim(self.left_win, pos=[0, -self.text_v_offset], 
                                          height = self.text_height)
        self.right_delay = visual.TextStim(self.right_win, pos=[0, -self.text_v_offset], 
                                           height = self.text_height)
        
        self.left_amount = visual.TextStim(self.left_win, pos=[0, self.text_v_offset],
                                           height = self.text_height)
        self.right_amount = visual.TextStim(self.right_win, pos=[0, self.text_v_offset],
                                            height = self.text_height)
        
        self.exp_info['subj_id'] = self.generate_subj_id()        
        self.exp_info['start_time'] = datetime.strftime(datetime.now(), '%b_%d_%Y_%H_%M')
        self.initialize_log()
        
    def show_test_screen(self, duration = 2.0):
        clock = core.Clock()
        while clock.getTime() < duration:
            self.left_test.draw()
            self.right_test.draw()
            
            self.left_win.flip()
            self.right_win.flip()
    
    def show_messages(self, left_message, right_message = None):
        if right_message is None:
            right_message = left_message
            
        event.clearEvents()
        while True:
            self.left_message.setText(left_message)
            self.right_message.setText(right_message)
            
            self.left_message.draw()
            self.right_message.draw()
            
            self.left_win.flip()
            self.right_win.flip()
            
            if event.getKeys(['space']):
                break
            elif event.getKeys(['escape']):
                core.quit()

    def show_choices(self, left_choice, right_choice):
        if self.auto_control:
            while not self.is_far_enough(self.get_current_position()):
                print(self.get_current_position())
                self.left_win.flip()
                self.right_win.flip()
                
        self.left_delay.setText(left_choice['delay'])
        self.right_delay.setText(right_choice['delay'])
        
        self.left_amount.setText(left_choice['amount'])
        self.right_amount.setText(right_choice['amount'])
               
        response = ''
        event.clearEvents()
            
        trial_start_time = time.clock()
        while True:
            self.left_delay.draw()
            self.right_delay.draw()
            self.left_amount.draw()
            self.right_amount.draw()
            
            self.left_win.flip()
            self.right_win.flip()            

            if self.auto_control:
                is_over, resp = self.is_near_monitor(self.get_current_position())
                if is_over:
                    response = resp
                    break
            else:
#            TODO: left and right don't work, check the problem here
                if event.getKeys(['lctrl']):
                    response = 'left'
                elif event.getKeys(['rctrl']):
                    response = 'right'
                elif event.getKeys(['space']):
                    break
                
            if event.getKeys(['escape']):
                core.quit()
        response_time = time.clock()-trial_start_time
        return response, response_time
    
    def run_exp(self):
        trial_params = self.get_trial_params()
        self.subj_id = self.generate_subj_id()
        n = self.n_trial
#        n = len(trial_params)
        for i in range(1, n+1):
            trial_info = self.run_trial(i, trial_params[i-1])
            self.write_trial_log(trial_info)
            
    
    def run_trial(self, trial_no, params):
        left_choice = {}
        right_choice = {}
        
        left_choice['delay'] = 'today' if params[0]==0 else 'in %d days'%params[0]
        left_choice['amount'] = u'\u20AC %d' % params[1]
        
        right_choice['delay'] = 'in %d days' % params[2]        
        right_choice['amount'] = u'\u20AC %d' % params[3]
        
        self.show_messages('Ready?')
        self.show_messages('Go!')
        self.show_messages(' ')
        response, response_time = self.show_choices(left_choice, right_choice)
        
        return [self.exp_info['subj_id'], trial_no,
                left_choice['delay'], left_choice['amount'], 
                right_choice['delay'], right_choice['amount'], 
                response, response_time]
    
    def get_trial_params(self):
        trial_params = np.loadtxt('mcq.csv', skiprows = 1, delimiter = ',')
        return trial_params
        
    def generate_subj_id(self):
        existing_subj_ids = np.loadtxt('existing_subj_ids.txt')
        subj_id = int(random.uniform(100, 999))
        while subj_id in existing_subj_ids:
            subj_id = int(random.uniform(100, 999))

        with open('existing_subj_ids.txt', 'ab+') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerow([str(subj_id)])
        return str(subj_id)

    def initialize_log(self, exp_info):
        self.log_file = 'data/' + exp_info['subj_id'] + '_' + exp_info['start_time'] + '.txt'    
        with open(self.log_name, 'ab+') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerow(['subj_id', 'trial_no', 'left_delay', 'left_amount', 
                   'right_delay', 'right_amount', 'response', 'RT'])
        
    def write_trial_log(self, trial_info):
        with open(self.log_file, 'ab+') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerow(trial_info)

    def get_current_position(self):
        log = np.loadtxt(self.trajectory_path, delimiter = ' ')
        return (log[-1,[1,2]] + log[-1,[5,6]])/2
        
    def is_near_monitor(self, position):
        if np.sqrt((position[0]-self.left_resp_area_center[0])^2 + \
            (position[1]-self.left_resp_area_center[1])^2) < self.resp_area_radius:
                return 'left', True
        elif np.sqrt((position[0]-self.right_resp_area_center[0])^2 + \
            (position[1]-self.right_resp_area_center[1])^2) < self.resp_area_radius:
                return 'right', True
        else:
            return None, False
    
    def is_far_enough(self, position):
        return True if position[1]>self.start_threshold else False
        
we = WalkingExp()
we.initialize_exp()
we.run_exp()

