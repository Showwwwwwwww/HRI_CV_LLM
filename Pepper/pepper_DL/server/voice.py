import qi
import argparse
import sys
import time
import os

class SpeechManager:
    def __init__(self, session):
        self.speech_service  = session.service("ALTextToSpeech")
        self.speech_service.setLanguage("English")

    def say(self, text, **kwargs):
        """ Makes Pepper say whatever text is
        Params:
            text: a string
                represents what you want Pepper to say
        """
        self.speech_service.say(text, **kwargs)

    def target_lost(self):
        self.speech_service.say("Target lost")

    def target_detected(self):
        self.speech_service.say("Target detected")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.137.8",
                        help="Robot IP address. O n robot or Local Naoqi: use '127.0.0.1'.")
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

    manager = SpeechManager(session)
    memory = session.service("ALMemory")
    manager.say("I'm Pepper, I like pineapple and anchovies on Pizza. Fight me")