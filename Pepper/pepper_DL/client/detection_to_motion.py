import numpy as np

class DetectionManager:
    def __init__(self, client, image_size):
        self.client = client
        self.image_size = image_size
        self.center = [dim/2 for dim in image_size]

    def center_x(self, detection, offset=0.05):
        """ Given bounding box data (x1,y1,x2,y2,), determine how much the robot should rotate to center x

        params:

        detection: [x1,y1,x2,y2]
            Detection data of the bounding box of interest
        offset: float
            Determines when to stop rotating. If offset is high, the robot will stop rotating if the target is not
            centered.
        """
        # Centered by rotating robot body
        center = (detection[0]+detection[2])/2
        diff = (self.center[0] - center)/self.center[0] # This is now a percentage
        if diff<-offset:
            # camera should move right
            # motion_service.walkToward(theta=diff)
            pass
        elif diff>offset:
            # camera should move left
            pass



    def center_y(self, detection, offset=0.05, up_bias=0):
        """ Given bounding box data (x1,y1,x2,y2,), determine how much the robot should rotate to center y

        params:

        detection: [x1,y1,x2,y2]
            Detection data of the bounding box of interest
        offset: float
            Determines when to stop rotating. If offset is high, the robot will stop rotating if the target is not
            centered.
        up_bias: float
            Determines how much the robot should focus above the center of the frame. Used to look at human faces.
        """
        # Centered by rotating robot head
        pass

    def hand_raised(self, detections):
        # indices of keypoints:
        # left_shoulder 5
        # left_wrist = 9
        # right_shoulder=6
        # right_wrist = 10

        pass

