import os
import argparse
import qi
import numpy as np
import time
import sys
import soundfile as sf

class AudioManager2(object):
    def __init__(self,session):
        super(AudioManager2,self).__init__()      
        self.module_name = "AudioManager2"
        # Get the services
        self.audio_service =session.service("ALAudioDevice")
        # Enable energy input compution 
        self.audio_service.enableEnergyComputation()
        # Audio recording setting
        self.threshold = 1000
        self.silence_duration = 3
        self.is_recording = False
        self.last_sound_time = None
        self.sample_rate = 16000
        #self.channels = [0,0,1,0]  # use front microphone
        self.channels = 3  # use front microphone
        self.isProcessingDone = False
        # Audio file setting 
        self.tmppath = ""
        self.wavfile = self.tmppath + "recording.wav"
        self.rawfile = self.tmppath + "rawrecording.raw"
        self.rawoutput = open(self.rawfile, "wb+")
        self.framesCount = 0
        self.nbOfFramesToProcess =100
        
    
    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        """
        Record the audio data from the front microphone depend on the sound threshold
        """
        # keep recording until reach the set frame number
        self.framesCount = self.framesCount + 1
        #print(self.framesCount)
        # if (self.framesCount <= self.nbOfFramesToProcess):
        #     print(self.framesCount)
        #     print(self.isProcessingDone)
        #     self.rawoutput.write(inputBuffer)
        # else:
        #     #print(self.isProcessingDone)
        #     self.isProcessingDone = True
        #     self.rawoutput.close()
        #Record wav with sound threshold
        front_energy = self.audio_service.getFrontMicEnergy()
        #print("Front energy: ", front_energy)
        if front_energy > self.threshold:
            print("Front energy: ", front_energy)
            self.rawoutput.write(inputBuffer)
            self.last_sound_time = time.time()
        elif self.last_sound_time is not None and (time.time() - self.last_sound_time) > self.silence_duration:
            print("Silence")
            self.isProcessingDone = True
            self.rawoutput.close()

    def startProcessing(self):
        """
        Subscribe the service and return the audio data
        """
        # try:
        self.audio_service.setClientPreferences(self.module_name, self.sample_rate, self.channels, 0)
        self.audio_service.subscribe(self.module_name)
        while self.isProcessingDone == False:
            time.sleep(1)

        self.audio_service.unsubscribe(self.module_name)
    # except Exception as e:
        #     print("Error while subscribing", e)
        # finally:
        #     print("Closing")
        #     self.audio_service.unsubscribe(self.module_name)
        # Read the data from the raw file and save it as a wav file
        data, samplerate = sf.read(self.rawfile, channels=1, samplerate=16000, subtype='PCM_16')
        sf.write(self.wavfile, data, samplerate)
        print "The recording is saved here: " + self.wavfile
        
        #return self.wavfile
        return data, samplerate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.0.52",
                        help="Robot IP address. On robot or Local Naoqi: use '192.168.137.26'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    args = parser.parse_args()

    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["AudioManager2", "--qi-url=" + connection_url])
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
                                                                                              "Please check your "
                                                                                              "script arguments. Run "
                                                                                              "with -h option for "
                                                                                              "help.")
        sys.exit(1)

    app.start() 
    MyAudioManager = AudioManager2(app.session)
    print("audioManager.module_name:" + MyAudioManager.module_name)
    #MyAudioManager.audio_service.unsubscribe(MyAudioManager.module_name)
    app.session.registerService(MyAudioManager.module_name, MyAudioManager)
    # Start processing the audio data
    print("Start processing the audio data...")
    audio_data = MyAudioManager.startProcessing()
    print("Audio data: ", audio_data)
