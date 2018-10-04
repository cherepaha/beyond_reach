from pykinect.nui import JointId

# this is used to calculate the average position, which is used for 
# checking responses, deadzone, etc.
POSITION_JOINTS = (JointId.Spine,         
                   JointId.HipCenter,                            
                   JointId.HipLeft, 
                   JointId.HipRight)

# lists all the joints which are tracked
# positions of these joints are recorded alongside the average position
TRACKED_JOINTS = (JointId.HipCenter,
                  JointId.Spine,                  
                  JointId.ShoulderCenter,
                  JointId.Head,
                  JointId.ShoulderLeft,
                  JointId.ElbowLeft, 
                  JointId.WristLeft, 
                  JointId.HandLeft,
                  JointId.ShoulderRight,
                  JointId.ElbowRight, 
                  JointId.WristRight, 
                  JointId.HandRight,
                  JointId.HipLeft, 
                  JointId.KneeLeft, 
                  JointId.AnkleLeft, 
                  JointId.FootLeft,             
                  JointId.HipRight, 
                  JointId.KneeRight, 
                  JointId.AnkleRight, 
                  JointId.FootRight)

# Logitech slide switcher is used to control the flow of the walking task. The subjects are 
# instructed to press the 'next slide' button to proceed through the information screens, 
# and to indicate their choice
KEYLIST = ['pagedown']
# other available buttons are: 'pageup', 'f5', 'period'

SCREEN_SIZE = (1920, 1080)

