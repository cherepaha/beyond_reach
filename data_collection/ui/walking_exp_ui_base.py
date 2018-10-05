import random
import numpy as np
from psychopy import visual, core
from win32api import GetSystemMetrics

class WalkingExpUIBase(object):
    currency_symbol = u'\u00A5' #% JPY    

    def __init__(self):
        self.screen_size = (GetSystemMetrics(0), GetSystemMetrics(1))
        self.is_ss_left = random.choice([True, False])
        print('Smaller sooner displayed on the left? %r' % (self.is_ss_left))
        
        self.left_message = visual.TextStim(self.left_win, color='#FFFFFF', 
                                            height=self.main_text_height)
        self.right_message = visual.TextStim(self.right_win, color='#FFFFFF', 
                                             height=self.main_text_height)
        self.left_message.setAutoDraw(False)
        self.right_message.setAutoDraw(False)
        
        self.left_delay = visual.TextStim(self.left_win, color='#FFFFFF',
                                          pos=self.left_delay_pos, 
                                          height=self.main_text_height)
        self.right_delay = visual.TextStim(self.right_win, color='#FFFFFF',
                                           pos=self.right_delay_pos, 
                                           height=self.main_text_height)
        
        self.left_amount = visual.TextStim(self.left_win, color='#FFFFFF',
                                           pos=self.left_amount_pos,
                                           height=self.main_text_height)
        self.right_amount = visual.TextStim(self.right_win, color='#FFFFFF',
                                            pos=self.right_amount_pos,
                                            height=self.main_text_height)
        
    def show_messages(self, left_message, right_message=None):            
        self.left_message.setText(left_message)
        self.left_message.draw()
        self.left_win.flip()

        if right_message is None:
            right_message = left_message
        
        if not (self.right_win == self.left_win):
            self.right_message.setText(right_message)
            self.right_message.draw()
            self.right_win.flip()

    def show_choices(self, params, trial_info):
        ss_choice = {}
        ll_choice = {}

        ss_choice['delay'] = 'today' if params[0]==0 else 'in %d days' % (params[0])        
        ss_choice['amount'] = self.currency_symbol + '{:,}'.format(int(params[1])).replace(',',' ')
        
        ll_choice['delay'] = 'in %d days' % params[2]        
        ll_choice['amount'] = self.currency_symbol + '{:,}'.format(int(params[3])).replace(',',' ')
        
        if self.is_ss_left:
            left_choice = ss_choice
            right_choice = ll_choice
        else:
            left_choice = ll_choice
            right_choice = ss_choice

        self.left_delay.setText(left_choice['delay'])
        self.right_delay.setText(right_choice['delay'])
        
        self.left_amount.setText(left_choice['amount'])
        self.right_amount.setText(right_choice['amount'])

        response = None
        response_dynamics_log = []     
        
        trial_start_time = core.getTime()

        # in some kinect trials subjects disappear from the field of view before they get to
        # the deadzone, which means the first sampled position from kinect is outside of deadzone
        # even if the subject is in the deadzone
        # to avoid this, in the beginning of each trial the position is set to 0 before the loop
        position = (0, 0)
        while self.is_in_deadzone(position):
            position = self.get_position()
            self.flip_screens()
            print(str(position[:2]) + ": in deadzone? " + str(self.is_in_deadzone(position)))
        
        while response is None:           
            self.draw_choices()
            self.flip_screens()
            t = core.getTime() - trial_start_time
            position = self.get_position()     
            
            response_dynamics_log.append(([trial_info['subj_id'], trial_info['trial_no'], '%.3f' % (t)] + 
                                           list(['%.4f' % value for value in position])))
            
            response = self.get_response()
            print(position[:2])
            print(response)
   
        print(response, self.is_ss_left)
        
        return response, response_dynamics_log 
    
    def draw_choices(self):
        self.left_delay.draw()
        self.right_delay.draw()
        self.left_amount.draw()
        self.right_amount.draw()
    
    def show_trial_start_screen(self):
        pass
    
    def flip_screens(self):
        self.left_win.flip()
        if not (self.right_win is self.left_win):
            self.right_win.flip()
            
    def get_response(self):
        '''
        MUST BE IMPLEMENTED IN AN INHERITING CLASS
        Returns 'left' or 'right' if the decision is reported and 'None' otherwise
        '''
        pass
        
    def get_position(self):
        '''
        MUST BE IMPLEMENTED IN AN INHERITING CLASS
        Returns the position of the decision maker
        The position[0] and position[1] should be the main x- and y-coordinates
        Other elements may contain extra dimensions of the position        
        '''
        pass
        
    def is_in_deadzone(self, position):       
        # if nan or is in actual deadzone, return True
        if ((np.isnan(position[0]) | np.isnan(position[1])) |
            ((position[0] > self.deadzone[0][0]) & (position[0] < self.deadzone[0][1]) &
            (position[1] > self.deadzone[1][0]) & (position[1] < self.deadzone[1][1]))):
            return True
        else: return False
            