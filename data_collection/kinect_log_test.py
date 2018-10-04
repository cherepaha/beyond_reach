import time
import numpy as np
from pykinect import nui

import constants

kinect = nui.Runtime()
kinect.skeleton_engine.enabled = True

def post_frame(frame):
    skeleton_positions = None
    for skeleton in frame.SkeletonData:
        # NB: 2 corresponds to the 'Tracked' status
        if (skeleton.eTrackingState == 2):            
            skeleton_positions = skeleton.SkeletonPositions
    
    mean_x, mean_y = (0,0)
    
    if skeleton_positions:
        x = np.array(list(-skeleton_positions[joint].x for joint in constants.POSITION_JOINTS))
        mean_x = np.mean(x[x.nonzero()])
        
        y = np.array(list(skeleton_positions[joint].z for joint in constants.POSITION_JOINTS))
        mean_y = np.mean(y[y.nonzero()])
        
    print(mean_x, mean_y)
#    for i, skeleton in enumerate(frame.SkeletonData):
#        if (skeleton.eTrackingState == 2):            
#            print(i)
##            print(skeleton.Position)
#            for joint in constants.POSITION_JOINTS:
#                print(joint)
#                print(skeleton.SkeletonPositions[joint].x, 
#                      skeleton.SkeletonPositions[joint].y,
#                      skeleton.SkeletonPositions[joint].z)            
    time.sleep(0.2)

kinect.skeleton_frame_ready += post_frame

done = False

while not done:
    continue