import qi
import argparse
import sys
import time

class EyeLEDManager:
    def __init__(self, session):
        self.leds_service = session.service("ALLeds")

        # Define the LED groups for the right and left eyes
        self.right_eye_leds = [
            "FaceLedRight0", "FaceLedRight1", "FaceLedRight2", "FaceLedRight3",
            "FaceLedRight4", "FaceLedRight5", "FaceLedRight6", "FaceLedRight7"
        ]
        self.left_eye_leds = [
            "FaceLedLeft0", "FaceLedLeft1", "FaceLedLeft2", "FaceLedLeft3",
            "FaceLedLeft4", "FaceLedLeft5", "FaceLedLeft6", "FaceLedLeft7"
        ]

    def set_eye_color(self, color):
        """
        Sets the color for all LEDs in the eyes
        Params:
            color: int
                The RGB value of the color to set the LEDs to, in hexadecimal format (e.g., 0x00FF0000 for red)
        """
        for led in self.right_eye_leds + self.left_eye_leds:
            self.leds_service.fadeRGB(led, color, 0.1)  # Duration is set to a small value for immediate change

    def set_eyes_blue(self):
        """
        Sets the eye LEDs to blue
        """
        blue_color = 0x000000FF
        self.set_eye_color(blue_color)

    def set_eyes_green(self):
        """
        Sets the eye LEDs to green
        :return:
        """
        green_color = 0x0000FF00
        self.set_eye_color(green_color)
        
    def set_eyes_red(self):
        """
        Sets the eye LEDs to red
        """
        red_color = 0x00FF0000
        self.set_eye_color(red_color)

    def turn_off_eyes(self):
        """
        Turns off all the eye LEDs
        """
        for led in self.right_eye_leds + self.left_eye_leds:
            self.leds_service.off(led)

    def __del__(self):
        """
        Destructor to ensure LEDs are turned off when the object is deleted
        """
        self.turn_off_eyes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.0.52",
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

    manager = EyeLEDManager(session)

    # Example usage
    manager.set_eyes_blue()
    print("Eyes set to blue. Press Enter to change to red.")
    time.sleep(5)  # Wait for user input to change color
    manager.set_eyes_red()
    print("Eyes set to red. Press Enter to exit.")
    time.sleep(5)  # Wait for user input to exit

    # Clean up
    del manager
