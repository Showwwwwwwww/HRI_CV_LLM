import qi
import argparse
import sys
import time
import os
from PIL import Image

class CameraManager():
    def __init__(self, session, resolution=5, colorspace=11, fps=5):
        self.camera_service = session.service("ALVideoDevice")
        #self.video_client = self.camera_service.subscribe("cam", resolution, colorspace, fps)
        self.video_client = self.camera_service.subscribeCamera("cam", 0, resolution, colorspace, fps)

        # Set all parameters to default
        self.camera_service.setAllParametersToDefault(0)
        #self.camera_service.setParameterToDefault(0, 8)
        #self.camera_service.setParameterToDefault(0,7)

        # Set parameters

        # Can't use parameter names for some reason, but these numbers mean:
        # position 0: CameraIndex = 0 # Use the top camera
        # position 1: Parameter = 8 # Refers to vertical flip
        # position 3: newValue = 1 # Default image is upsidedown, so setting this param to 1 will flip it back
        self.camera_service.setParameter(0, 8, 1) # Vertical flip
        #self.camera_service.setParameter(0, 7, 1) # Horizontal flip

    def __del__(self):
        # Unsubbing
        self.camera_service.unsubscribe(self.video_client)

    def get_image(self, raw=True, save_dir=None, img_name=None):
        img = self.camera_service.getImageRemote(self.video_client)
        self.camera_service.releaseImage(self.video_client)
        if not raw:
            img = self.convert_to_pillow(img)
            if save_dir is not None and img_name is not None:
                self.save(img, img_name, save_dir)
        return img


    def convert_to_pillow(self, img, rotate=0):
        #start = time.time()
        #img = self.get_image()
        real_img = Image.frombuffer('RGB', (img[0], img[1]), bytes(img[6]))
        #real_img.show()
        return real_img.rotate(rotate)
        #end = time.time()
        #print "It took " + str(end-start) + " seconds to take a photo and display it"

    def save(self, img, img_name, save_dir="photos"):
        # Assumed that img is already a Pillow Image object
        img.save(os.path.join(save_dir, img_name))

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
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
                                                                                             "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    manager = CameraManager(session, resolution=1, colorspace=11, fps=30)
    #raw_image = manager.get_image()
    #image = manager.convert_to_pillow(raw_image)

    #image.show()
    """
    try:
        while True:
            start = time.time()
            img = manager.get_image()
            #real_img = Image.frombuffer('RGB', (img[0], img[1]), bytes(img[6]))
            real_img = manager.convert_to_pillow(img)
            end = time.time() - start
            print "It took " + str(start-end) + " to take a photo and display it"
            #real_img.show()
            #time.sleep(1)
    except KeyboardInterrupt:
        print
        print "Interrupted by user"
        print "Stopping..."


    del manager

    """
    start = time.time()
    frames = 60
    for _ in range(frames):
        img = manager.get_image(raw=False)
        # real_img = Image.frombuffer('RGB', (img[0], img[1]), bytes(img[6]))
    end = time.time() - start
    print "It took " + str(end) + " to take " + str(frames)  + " photos and send to server, achieving " + str(frames/end) + "FPS."