# A dialog system for SoftBank's Pepper robot using Google ASR and DialogFlow services

# It records sound from Pepper's front microphone (can be set to using all 4 microphones instead).
# The recording is saved locally as a wav format that Google understands, then sent to Google ASR to get transcribed. 
# The transcript is sent to Google DialogFlow. The received reponse is then spoken by Pepper.

# You will need a DialogFlow API key: https://dialogflow.com/docs/reference/v2-auth-setup
# You can either define your own QA pairs or make use of the pre-built DialogFlow agents: https://dialogflow.com/docs/agents#prebuilt_agents

# To do: dynamic recording length based on silent pause duration
# To do: multi-turn instead of single turn


__version__ = "0.1.0"

__copyright__ = "Copyright 2018, Monash University"
__author__ = 'Leimin Tian'
__email__ = 'Leimin.Tian@monash.edu'


import argparse
import sys
import os
import time
import json
import array
import numpy as np
# Install these libraries in your local python, or upload the packages to Pepper
import qi
import soundfile as sf
import speech_recognition as sr
import apiai


class DialogModule(object):
    def __init__(self, app):
        """
        Initialise services and variables.
        """
        super(DialogModule, self).__init__()
        app.start()
        session = app.session

        # Get the services
        self.audio_service = session.service("ALAudioDevice")
        self.tts_service = session.service("ALTextToSpeech")
        
        # initialize parameters
        self.isProcessingDone = False
        self.nbOfFramesToProcess = 60 # fixed length recording at the moment
        self.framesCount=0
        self.tmppath = ""
        self.wavfile = self.tmppath + "recording.wav"
        self.rawfile = self.tmppath + "rawrecording.raw"
        self.rawoutput = open(self.rawfile,"wb+")
        self.module_name = "DialogModule"        

    # keep track of the frame counts
    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        # keep recording until reach the set frame number
        self.framesCount = self.framesCount + 1
        if (self.framesCount <= self.nbOfFramesToProcess):
            self.rawoutput.write(inputBuffer)
        else :
            self.isProcessingDone=True
            self.rawoutput.close()  
                
    # record wav and do Google ASR
    def ASR(self):                
        # ask for the front microphone signal sampled at 16kHz
        # if you want the 4 channels call setClientPreferences(self.module_name, 48000, 0, 0)
        self.audio_service.setClientPreferences(self.module_name, 16000, 3, 0)
        self.audio_service.subscribe(self.module_name)        

        while self.isProcessingDone == False:
            time.sleep(1)

        self.audio_service.unsubscribe(self.module_name) 

        # record in a wav format that Google ASR understands
        data, samplerate = sf.read(self.rawfile, channels=1, samplerate=16000, subtype='PCM_16')
        sf.write(self.wavfile, data, samplerate)
        print "The recording is saved here: " + self.wavfile

        # Google ASR
        r = sr.Recognizer()
        audiofile = sr.AudioFile(self.wavfile)

        with audiofile as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)

        try:
            asr = r.recognize_google(audio)
            print "This is what Pepper thinks you said: " + asr
        except sr.UnknownValueError:
            asr = ""
            print "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            asr = ""
            print "Could not request results from Google Speech Recognition service; {0}".format(e)
            
        return asr

    # pass on asr transcript to Google DialogFlow and receive the response
    def DialogFlow(self, asr):
        # use your Dialog Flow API key here
        CLIENT_ACCESS_TOKEN = '0123456789abcdefghijklmnopqrstuvwxyz'
        ai = apiai.ApiAI(CLIENT_ACCESS_TOKEN)
        request = ai.text_request()
        request.lang = 'en'  # optional, default value equal 'en'
        request.session_id = "1"
        request.query = asr
        
        # receive response from DialogFlow
        if asr == "":
            reply = "Sorry, I didnt catch what you said."
        else:
            response = request.getresponse()
            r = response.read()     #can be obtained only once
            json_dict = json.loads(r)
            r2 = format(json_dict['result']['fulfillment']['speech'])
            r3 = r2.encode('utf-8') 
            if r3 == "":
                reply = "Sorry, I dont know how to answer that."
            else:
                reply = r3
        print "This is Pepper's answer to you: " + reply
        
        return reply
    
    # Pepper text-to-speech
    def TTS(self, reply):
        self.tts_service.setParameter("speed", 80)# set speech speed to 80%
        self.tts_service.say(reply) # text-to-speech
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["DialogModule", "--qi-url=" + connection_url])        
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    
    MyDialogModule = DialogModule(app)
    app.session.registerService("DialogModule", MyDialogModule)
    # ASR
    print "Start listening..."
    asr = MyDialogModule.ASR()
    # Dialog Flow
    reply = MyDialogModule.DialogFlow(asr)
    # Text-to-speech
    MyDialogModule.TTS(reply)