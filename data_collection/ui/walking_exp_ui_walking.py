from pykinect import nui
from pykinect.nui import JointId
from psychopy import visual, event
from walking_exp_ui_base import WalkingExpUIBase
import numpy as np
import constants

class WalkingExpUIWalking(WalkingExpUIBase):    
    def __init__(self):
        self.left_win = visual.Window(size=constants.SCREEN_SIZE, units='norm', fullscr=True, color='#000000', screen=1) 
        self.right_win = visual.Window(size=constants.SCREEN_SIZE, units='norm', fullscr=True, color='#000000', screen=2)
                                       
        self.kinect = nui.Runtime()
        self.kinect.skeleton_engine.enabled = True
        self.kinect.skeleton_frame_ready += self.update_positions
        
        self.position = [np.nan, np.nan]
        self.deadzone  = ((-1, 1), (0, 1.2))
                                       
        self.main_text_height = 0.3
        self.text_h_offset = 0.0
        self.text_v_offset_left = 0.35
        self.text_v_offset_right = 0.3
        
        self.main_text_height_left = 0.35
        self.main_text_height_right = 0.2
        
        self.response_x_threshold = 0.8
        self.response_y_threshold = 3.5
                      
        self.left_amount_pos = (self.text_h_offset, self.text_v_offset_right)
        self.left_delay_pos = (self.text_h_offset, -self.text_v_offset_right)
        
        self.right_amount_pos = (self.text_h_offset, self.text_v_offset_right)
        self.right_delay_pos = (self.text_h_offset, -self.text_v_offset_right)
        
        super(WalkingExpUIWalking, self).__init__()

    def update_positions(self, frame):
        '''
        NB: in the Kinect coordinate frame, x goes from right to left (facing away from the 
        camera). For this reason, the sign of the x coordinate is changed right away. 
        NB: also, in the Kinect coordinate frame, y corresponds to height, and z to depth (distance 
        from the camera). As we are treating distance from the camera similarly to the vertical 
        coordinate of the mouse cursor, z-data is used to calculate y position.        
        '''
        self.position = np.repeat(np.nan, 2 + 3*len(constants.TRACKED_JOINTS))
        skeleton_positions = None
        
        for skeleton in frame.SkeletonData:
            # NB: 2 corresponds to the 'Tracked' status
            if (skeleton.eTrackingState == 2):            
                skeleton_positions = skeleton.SkeletonPositions
        
        if skeleton_positions:
            joint_positions = np.array(list([-skeleton_positions[joint].x, 
                                             skeleton_positions[joint].y, 
                                             skeleton_positions[joint].z]  
                                        for joint in constants.TRACKED_JOINTS)).flatten()
            
            x = np.array(list(-skeleton_positions[joint].x for joint in constants.POSITION_JOINTS))
            mean_x = np.mean(x[x.nonzero()])
            
            y = np.array(list(skeleton_positions[joint].z for joint in constants.POSITION_JOINTS))
            mean_y = np.mean(y[y.nonzero()])

            self.position = np.append([mean_x, mean_y], joint_positions)
    
    def show_messages(self, left_message, right_message=None):
        super(WalkingExpUIWalking, self).show_messages(left_message, right_message)
        
        event.waitKeys(keyList=constants.KEYLIST, maxWait=600)
    
    def get_response(self):
        keys = event.getKeys(keyList=constants.KEYLIST)
        if keys:
            position = self.get_position()
            if ((abs(position[0]) > self.response_x_threshold) & 
                (abs(position[1]) > self.response_y_threshold)):    
                return 'left' if position[0]<0 else 'right'
        else: return None
    
    def get_position(self): 
        return self.position
    
    def show_trial_start_screen(self):
        # this is an empty screen, possibly with info message, where the subject is supposed 
        # to click when ready to start
        super(WalkingExpUIWalking, self).show_trial_start_screen()
        
        self.show_messages('Go to start and click when ready')
    