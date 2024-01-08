import qi
import argparse
import sys
import time
import os

class MovementManager:
    def __init__(self, session):
        self.motion_service  = session.service("ALMotion")
        self.posture_service = session.service("ALRobotPosture")
        self.motion_service.setStiffnesses("Head", 1.0)
        # First, wake up.
        self.motion_service.wakeUp()
        self.reset_posture()
        # minimize Security Distance
        self.motion_service.setTangentialSecurityDistance(0.05)
        self.motion_service.setOrthogonalSecurityDistance(0.10)

    def reset_posture(self, speed=0.5):
        """ Makes the robot stand up straight. Used to reset posture.
        Params:

        speed: float
            Influences the speed in which the robot adopts the specified default posture. Ranges from 0 to 1.
        """
        # Wake up robot
        self.motion_service.wakeUp()
        # Make the robot stand up straight.
        self.posture_service.goToPosture("StandInit", speed)

    def walkTo(self, x=0, y=0, theta=0, verbose=False):
        """ Makes the robot walk at speeds relative to its maximum speed until its collision avoidance or the user
            causes it to stop. This function will not interrupt the main flow of the program, however calling this
            function multiple times will terminate older executions.

        Params:

        x: float
            Controls the speed in which the robot moves foward. Ranges from -1 to 1.
        y: float
            Controls the speed in which the robot strafes left. Ranges from -1 to 1.
        theta: float
            Controls the speed in which the robot rotates anti-clockwise. Ranges from -3.1415 to 3.1415.
        verbose: bool
            Determines whether we want to print out parameters and results.
        """
        if verbose:
            print "Walking with forward=", x, "m, left=", y, "m, rotation=", theta, "degrees anti-clockwise..."
        return self.motion_service.moveTo(x,y,theta)

    def walkToward(self, x=0, y=0, theta=0, verbose=False):
        """ Makes the robot walk at speeds relative to its maximum speed until its collision avoidance or the user
            causes it to stop. This function will not interrupt the main flow of the program, however calling this
            function multiple times will terminate older executions.

        Params:

        x: float
            Controls the speed in which the robot moves foward. Ranges from -1 to 1.
        y: float
            Controls the speed in which the robot strafes left. Ranges from -1 to 1.
        theta: float
            Controls the speed in which the robot rotates anti-clockwise.
        verbose: bool
            Determines whether we want to print out parameters and results.
        """
        if verbose:
            print "Walking with forward=", x, "m/s, left=", y, "m/s, rotation=", theta, "degrees anti-clockwise..."
        self.motion_service.moveToward(x,y,theta)

    def stop(self):
        """ Stops walking
        """
        self.motion_service.stopMove()

    def rotate_head_abs(self, forward=0, left=0, speed=0.2):
        """ Rotates Pepper's head relative to absolute position
        Params:
            forward: float
                Controls the extent of moving Pepper's head forward. Ranges from -2.0857 to 2.0857.
                This an absolute position, meaning each float corresponds to a particular location.
                Setting a number outside the range will set it to the maximum or minimum depending on the input.
            left: float
                Controls the extent of moving Pepper's head horizontally. Ranges from -0.7068 to 0.4451
            speed: float between 0 and 1
                Determines how fast the head rotation is. I'd advise setting it below 0.5 because anything above that
                    doesn't look healthy for the robot's hardware.
        """
        # HeadYaw = horizontal rotation
        # HeadPitch = vertical rotation
        self.motion_service.setAngles(["HeadYaw", "HeadPitch"], [left,forward], speed)

    def rotate_head(self, forward=0, left=0, speed=0.2):
        """ Rotates Pepper's head relative to current head position
        Params:
            forward: float
                Controls the extent of moving Pepper's head forward. Ranges from -2.0857 to 2.0857.
                This a relative position, meaning each float corresponds to the current location.
                Positive means look down, negative means look up.
                Setting a number outside the range will set it to the maximum or minimum depending on the input.
            left: float
                Controls the extent of moving Pepper's head horizontally. Ranges from -0.7068 to 0.4451
            speed: float between 0 and 1
                Determines how fast the head rotation is. I'd advise setting it below 0.5 because anything above that
                    doesn't look healthy for the robot's hardware.
        """
        self.motion_service.changeAngles(["HeadYaw", "HeadPitch"], [left,forward], speed)

    def get_head_position(self, rounded_deci=-1):
        """
        Params:
            rounded_deci: int
                Determines the number of decimal places to round to. If negative, no rounding will be done

        Returns:
            a list of length 2 in the format: [HeadYaw, HeadPitch]
        """
        pos = self.motion_service.getAngles(["HeadYaw", "HeadPitch"], True)
        return pos if rounded_deci < 0 else [round(x, rounded_deci) for x in pos]

    def rotate_body_to_head(self, speed = 0.2):
        """ Rotates the body to match with the head in terms of its horizontal rotation
        Params:
            speed: float
                Determines how fast this process will take
        """
        pass

    def terminate(self, fractionMaxSpeed=0.2):
        # Shuts down services upon deletion
        self.posture_service.goToPosture("Sit", fractionMaxSpeed)
        self.motion_service.rest()

    def __del__(self):
        # Shuts down services upon deletion
        self.posture_service.goToPosture("Sit", 0.2)
        self.motion_service.rest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.137.131",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
                                                                                             "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    manager = MovementManager(session)
    memory = session.service("ALMemory")
    # del manager