import os
import argparse
import qi
import numpy as np
import time
import sys

app = qi.Application()
qicore = qi.module('qicore')

class AudioManager(object):

    def __init__(self, session):
        self.session = session
        self.audio_service = session.service("ALAudioDevice")
        self.record_service = session.service("ALAudioRecorder")
        #self.audio_player = session.service("ALAudioPlayer")
        self.threshold = 500
        self.silence_duration = 3
        self.is_recording = False
        self.last_sound_time = None
        self.sample_rate = 16000
        self.channels = [0,0,1,0]  # use front microphone
        #self.record_path= './../database/record.wav'
        self.record_path = '/home/nao/record.wav'
        self.audio_subscription = None
        self.playAudio = False
        self.finished = False   
        self.audio_content = None

        # Enable energy computation for get the energy(volume) of the front microphone
        self.audio_service.enableEnergyComputation()

        
    def setPlayAudio(self, playAudio):
        self.playAudio = playAudio

    def start_recording(self):
        if self.is_recording:
            return
        try:
            self.record_service.startMicrophonesRecording(self.record_path, "wav", self.sample_rate, self.channels)
        except RuntimeError as e:
            # print(f"Already recording, attempting to restart")
            self.record_service.stopMicrophonesRecording()
            self.record_service.startMicrophonesRecording(self.record_path, "wav", self.sample_rate, self.channels)

        self.is_recording = True
        print("Recording started:", self.record_path)
    
    def test_write_path(self,path):
        try:
            with open(os.path.join(path, 'test_file.txt'), 'w') as test_file:
                test_file.write('Hello, World!')
            os.remove(os.path.join(path, 'test_file.txt'))
            print("Write test successful.")
        except Exception as e:
            print("Write test failed:", str(e))

    def stop_recording(self, download = False):
        if not self.is_recording:
            return 
        self.record_service.stopMicrophonesRecording()
        self.is_recording = False
        print("Recording stopped:", self.record_path)
        self.finished = True
        if self.playAudio and not self.is_recording:  # Play the audio and recorded finished
            self.play_audio()
        # ### Directory exist testing
        # current_directory = os.getcwd()
        # print("Current Working Directory:", current_directory)  
        # self.test_write_path('/home/nao/')
        time.sleep(3)  # Wait for 2 seconds to ensure the file system updates
        if download:
            try:
                file = qicore.openLocalFile(self.record_path)
                self.audio_content = file.read()
                del file
            except Exception as e:
                print("Error reading file:", str(e))
        
    def play_audio(self):
        print("Playing audio file:", self.record_path)
        self.audio_service.playFile(self.record_path,1,0)

    def check_mic_energy(self):
        energy = self.audio_service.getFrontMicEnergy()
        print("Current front mic energy:", energy)
        if energy > self.threshold:
            if not self.is_recording:
                self.start_recording()
            self.last_sound_time = time.time()
        elif self.is_recording and (time.time() - self.last_sound_time) >= self.silence_duration:
            self.stop_recording()
            self.finished = True


    def __del__(self):
        if self.audio_subscription is not None:
            self.audio_service.unsubscribe(self.audio_subscription)
 
    def get_recording(self):
        print("Recording audio...")
        while not self.finished:
            self.check_mic_energy()
            time.sleep(0.1)  # check every 0.1 seconds
        #return self.record_path
        return self.audio_content
    def sample_recording(self):
        energy = self.audio_service.getFrontMicEnergy()
        print("Current front mic energy:", energy)

        self.record_service.startMicrophonesRecording(self.record_path, "wav", self.sample_rate, self.channels)
        time.sleep(10)
        self.record_service.stopMicrophonesRecording()
        print 'record over'

        fileId = self.audio_service.playFile(self.record_path, 1.0, 0.0)
        # print 'play over' + fileId
        
        file = qicore.openLocalFile(self.record_path)
        content = file.read()
        print(len(content))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.0.52", help="Robot IP address.")
    parser.add_argument("--port", type=int, default=9559, help="Naoqi port number.")
    args = parser.parse_args()

    # Connect to the robot
    # app = qi.Application(["AudioManager", "--qi-url=tcp://{}:{}".format(args.ip, args.port)])
    # app.start()
    session = qi.Session()
    #session = app.session
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
                                                                                             "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    audio_manager = AudioManager(session)
    # audio_manager.setPlayAudio(True)

    audio_manager.sample_recording()

    # try:
    #     while not audio_manager.finished:
    #         audio_manager.check_mic_energy()
    #         time.sleep(0.1)  # check every 0.1 seconds
    # except KeyboardInterrupt:
    #     print("Interrupted by user, shutting down")
    #     sys.exit(0)

    del audio_manager