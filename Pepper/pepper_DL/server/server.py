import socket
import os
import sys

import numpy as np
import qi
import cPickle
import time
import argparse
import io
import base64

from flask import Flask, request, Response, jsonify,send_file
from PIL import Image
from camera import CameraManager
from locomotion import MovementManager
from voice import SpeechManager
from audio import AudioManager


# Initiate Qi Session
session = qi.Session()
parser = argparse.ArgumentParser(description="Please enter Pepper's IP address (and optional port number)")
parser.add_argument("--ip", type=str, nargs='?', default="192.168.1.102")
# IP on router: 192.168.1.102
# IP on router: 169.254.220.108
parser.add_argument("--port", type=int, nargs='?', default=9559)
args = parser.parse_args()

print("Received IP: ", args.ip)
print("Received port: ", args.port)
# Connecting to the robot
# Decided not to use try-except because if the server failed to connect to the robot, then there's no point
    # starting a Flask server
session.connect("tcp://" + args.ip + ":" + str(args.port))

print("Connected to Pepper!")

# Robot setup:
# Controls autonomous life behaviour, mainly used to disable it
print("Subscribing to live service...")
life_service = session.service("ALAutonomousLife")
# Controls the robot's cameras
print("Subscribing to camera service...")
camera_manager = CameraManager(session, resolution=0, colorspace=11, fps=30)
# Controls the robot's speech
print("Subscribing to speech service...")
speech_manager = SpeechManager(session)
# Control the robot's microphone
audio_manager = AudioManager(session)
# Disable autonomous life because it can interfere with follow behaviour
life_service.setAutonomousAbilityEnabled("All", False)

# Start Flask server
app = Flask(__name__)

@app.route("/audio/get_latest_recording", methods=["GET"])
def get_latest_recording():
    audio_file_path = audio_manager.get_latest_recording()
    if audio_file_path and os.path.exists(audio_file_path):
        return send_file(audio_file_path, mimetype="audio/wav", as_attachment=True, attachment_filename=os.path.basename(audio_file_path))
    else:
        return jsonify({'error': 'No recording found'}), 404

@app.route("/image/send_image", methods=["POST"])
def send_image():
    # Requests image from Pepper, then sends it to the client
    img = camera_manager.get_image(raw=False) # Gets image as Pillow Image
    rawBytes = io.BytesIO()
    # Saves image as buffer
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({
        'msg': 'success',
        'img': str(img_base64),
    })


@app.route("/voice/say", methods=["POST"])
def say():
    # Makes Pepper say whatever based on what the client has sent over
    speech_manager.say(request.data)
    return jsonify({
        "msg": "success"})


@app.route("/setup/end", methods=["POST"])
def shutdown():
    # Run to shut down
    speech_manager.say("Shutting down")
    shutdown_server()
    return jsonify({
        'msg': 'success',
    })

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route("/test/pepper_to_server_fps", methods=["POST"])
def pepper_to_server_fps():
    start = time.time()
    frames = 60
    for _ in range(frames):
        img = camera_manager.get_image(raw=False)  # Gets image as Pillow Image
        rawBytes = io.BytesIO()
        # Saves image as buffer
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        img = np.array(Image.open(io.BytesIO(base64.b64decode(img_base64))))
    end = time.time() - start
    return jsonify({
        "time":str(end),
        "frames":str(frames)
    })
    print "It took " + str(end) + " seconds to send " + str(frames) + " frames at " + str(frames/end) + "FPS."


if __name__ == '__main__':
    # start flask app
    app.run(host="0.0.0.0", port=5000)
    del camera_manager
    del life_service
    del speech_manager
    del audio_manager
