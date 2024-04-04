import os
import argparse

import qi
import numpy as np
import time

class AudioManager():
    def __init__(self, session):
        self.session = session
        self.audio_service = session.service("ALAudioDevice")
        self.record_service = session.service("ALAudioRecorder")
        self.threshold = 2000
        self.silence_duration = 2
        self.is_recording = False
        self.last_sound_time = None
        self.sample_rate = 16000
        self.channels = [0, 0, 1, 0]  # Only use the front microphone
        self.dir_path = '/home/nao/audio/'
        self.audio_subscription = None

    def audio_callback(self, channels, samples, timestamp, buffer):
        pcm = np.frombuffer(buffer, dtype=np.int16)
        pcm = pcm.reshape((samples, channels))
        volume = np.max(np.abs(pcm))

        current_time = time.time()
        if volume > self.threshold:
            if not self.is_recording:
                self.start_recording()
            self.last_sound_time = current_time
        elif self.is_recording and current_time - self.last_sound_time >= self.silence_duration:
            self.stop_recording()

    def start_recording(self):
        if self.is_recording:
            return
        filename = "record_{}.wav".format(time.strftime("%Y%m%d-%H%M%S"))
        self.audio_file_path = os.path.join(self.dir_path, filename)
        self.record_service.startMicrophonesRecording(self.audio_file_path, "wav", self.sample_rate, self.channels)
        self.is_recording = True
        print("Recording started:", self.audio_file_path)

    def stop_recording(self):
        if not self.is_recording:
            return
        self.record_service.stopMicrophonesRecording()
        self.is_recording = False
        print("Recording stopped:", self.audio_file_path)

    def subscribe_to_audio(self):
        self.audio_subscription = self.audio_service.subscribe("PepperAudioHandler")
        self.audio_service.setClientPreferences(self.audio_subscription, self.sample_rate, self.channels, 0)
        # Connect the callback
        self.audio_service.setCallback(self.audio_callback)

    def __del__(self):
        if self.audio_subscription is not None:
            self.audio_service.unsubscribe(self.audio_subscription)

    def get_latest_recording(self):
        # check directory is exist or not
        if not os.path.exists(self.dir_path):
            print "Recording directory does not exist:"+self.dir_path
            return None

        # get all the file and sort by time
        files = [os.path.join(self.dir_path, f) for f in os.listdir(self.dir_path)]
        if not files:
            print "No recording files found in the directory:"+self.dir_path
            return None

        # the latest at the end, and re turn this file
        latest_file = max(files, key=os.path.getmtime)
        print "Latest recording file found:"+ latest_file
        return latest_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.43.183",
                        help="Robot IP address. On robot or Local Naoqi: use '192.168.137.26'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()

    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
                                                                                             "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    manager =AudioManager(session)

    # Assuming session is already created and connected to the robot
    audio_handler = AudioManager(session)
    audio_handler.subscribe_to_audio()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio_handler.unsubscribe_from_audio()

