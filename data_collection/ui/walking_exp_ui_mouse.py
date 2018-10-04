from psychopy import visual, event
from walking_exp_ui_base import WalkingExpUIBase
import constants

# TODO: move all constants to constants.py
class WalkingExpUIMouse(WalkingExpUIBase):
    def __init__(self):
        self.left_win = visual.Window(size=constants.SCREEN_SIZE, units='norm', 
                                      fullscr=True, screen=1, color='#000000') 
        self.right_win = self.left_win                                            
        self.mouse = event.Mouse(win=self.left_win, visible=True)        
        
        self.main_text_height = 0.1
        self.start_button_text_height = 0.05
        
        ### Parameters of the trial start screen ###
        self.start_button_pos = (0, -0.9)
        self.start_button_size = (0.15, 0.1)
        self.start_button_rect = visual.Rect(win=self.left_win, pos=self.start_button_pos, 
                                        width=self.start_button_size[0], 
                                        height=self.start_button_size[1],
                                        lineColor='#FFFFFF', fillColor=None, lineWidth=3)
                                                   
        self.start_button_text = visual.TextStim(self.left_win, text='Start', color='#FFFFFF',
                                                 pos=self.start_button_pos, 
                                                 height=self.start_button_text_height) 
        
        ### Parameters of the response screen ###    
        self.deadzone  = ((-1, 1), (-1, -0.7))        
        self.left_response_area_pos = (-0.42, 0.85)
        self.right_response_area_pos = (0.42, 0.85)
        
        
        # assuming 3:4 proportion between width and height, and 1.77:1 screen aspect ratio
        # we get the width of the response area in the PsychoPy normed units (-1 to 1) as
        # 2*(3/4)*(1/1.77)        
        self.response_area_size = (0.33, 0.25)
        
        self.text_h_offset = 0.85
        self.text_v_offset_left = 0.85
        self.text_v_offset_right = self.text_v_offset_left
        
        self.left_amount_pos = (self.left_response_area_pos[0], 
                                self.left_response_area_pos[1]+self.main_text_height/2)
        self.left_delay_pos = (self.left_response_area_pos[0], 
                               self.left_response_area_pos[1]-self.main_text_height/2) 

        self.right_amount_pos = (self.right_response_area_pos[0], 
                                 self.right_response_area_pos[1]+self.main_text_height/2)
        self.right_delay_pos = (self.right_response_area_pos[0], 
                                self.right_response_area_pos[1]-self.main_text_height/2)               
       
        self.left_response_area = visual.Rect(win=self.left_win, pos=self.left_response_area_pos,
                                              width=self.response_area_size[0], 
                                              height=self.response_area_size[1],
                                              lineColor='#FFFFFF', fillColor=None)
        self.right_response_area = visual.Rect(win=self.left_win, pos=self.right_response_area_pos, 
                                               width=self.response_area_size[0], 
                                               height=self.response_area_size[1],
                                               lineColor='#FFFFFF', fillColor=None)  
        
        super(WalkingExpUIMouse, self).__init__()
        
    def show_messages(self, left_message, right_message=None):
        super(WalkingExpUIMouse, self).show_messages(left_message, right_message)        
        # the first while loop is needed in case the mouse button is pressed at the time the message is shown
        # it waits until the mouse button is released
        while (self.mouse.getPressed()[0]):
            continue        
        # the second while loop waits until the mouse button is clicked
        while (not self.mouse.getPressed()[0]):
            continue            
    
    def get_response(self):
        if self.mouse.isPressedIn(self.left_response_area):
            return 'left'
        elif self.mouse.isPressedIn(self.right_response_area):
            return 'right'
        else: 
            return None
    
    def get_position(self):
        return self.mouse.getPos()
    
    def draw_choices(self):
        super(WalkingExpUIMouse, self).draw_choices()
        
        self.left_response_area.draw()
        self.right_response_area.draw()
        
    def show_trial_start_screen(self):        
        super(WalkingExpUIMouse, self).show_trial_start_screen()
        
        self.start_button_rect.draw()
        self.start_button_text.draw()
        self.flip_screens()
        
        # the first while loop is needed in case the mouse button is pressed at the time the screen is shown
        # it waits until the mouse button is released
        while (self.mouse.getPressed()[0]):
            continue
        
        # the second while loop waits until the start button is clicked
        while (not self.mouse.isPressedIn(self.start_button_rect)):
            continue  
    