import socket
import sys
import pickle
import numpy as np
import os
import cv2
import time
#import matplotlib
#matplotlib.use('TKAgg')
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from PIL import Image

# Uncomment the following lines to use SORT or OCSORT

# SORT:
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "sort"))
#from trackers.sort.sort import SortManager

# OCSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "ocsort"))
from trackers.ocsort.ocsort import OCSortManager
# Copied from: https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client

class Client:
    def __init__(self, port=5000, host=None, model="sort", **kwargs):
        self.host = socket.gethostname() #if host is not None else host
        self.port = port
        self.client_socket = socket.socket()
        print(f"Loading {model}...")
        #if model == "yolo":
        #    self.dl_model = YoloManager(**kwargs)
        #elif model == "sort":
        #    self.dl_model = SortManager(**kwargs)
        self.dl_model = OCSortManager(use_byte=True, **kwargs)
        print(model, " loaded successfully!")
        print(f"port: {self.port}")
        self.client_socket.connect((self.host, self.port))

    # Each model might require a unique function for configuration because they accept different parameters
    #def configure_yolo_model(self, weights='yoloposes_640_lite.pt', image_size=640, save_txt_tidle=True, device="cpu"):
    #    self.dl_model = YoloManager(weights=weights,  image_size=image_size, save_txt_tidle=save_txt_tidle, device=device)

    #def connect(self):
    #    self.client_socket.connect((self.host, self.port))
    def neo_communicate(self):
        while True:
            #fps = pickle.loads(self.client_socket.recv(1000), encoding="latin1")
            #duration = pickle.loads(self.client_socket.recv(1000), encoding="latin1")
            time.sleep(1)
            fps= None
            duration = False
            wait = 1/fps if fps is not None else 0
            if duration:
                count = 0
                start = time.time()
                t_end = start + duration
                frames = 0

                # Visualisation ini

                while time.time() < t_end:
                    data = self.client_socket.recv(300000)  # receive response
                    print("Received data is ", sys.getsizeof(data), " bytes.")
                    #print('Received from server: ' , data)  # show in terminal
                    data = pickle.loads(data, encoding='latin1')
                    real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                    img = np.asarray(real_img)[:,:,::-1]
                    print(img.shape)
                    #pred = self.dl_model.predict(img)

                    # Stream live data
                    cv2.imshow("stream", img)
                    cv2.waitKey(1)
                    self.send_text("a")
                    count += wait
                    frames += 1
                cv2.destroyAllWindows()
                end_time = time.time()-start
                print("FPS test for input=80x60 resolution wireless:")
                print("It took ", end_time, " seconds to stream ", frames, " frames.")
                #print("count: ", count)
                print("fps: ", fps)
                #print("duration: ", duration)
                print("true fps: ", frames/end_time)

            else:
                while True:
                    try:
                        img = self.receive_image(verbose=1)
                        #pred = self.dl_model.update(img)
                        pred, l = self.dl_model.smart_update(img)
                        # Shape of pred: number of tracked targets x 5
                        # where 5 represents: (x1, y1, x2, y2, id)

                        self.dl_model.draw(pred, np.ascontiguousarray(img), show=1)

                        # For capturing predictions
                        #self.dl_model.draw(pred, np.ascontiguousarray(img), show=1,save_dir=os.path.join("pepper_test", "rotate"))
                        m = self.center_target(pred, img.shape, vertical_offset=0.8, lost = l)
                        print("m = ", m)
                        self.send_text(m)
                    except KeyboardInterrupt:
                        self.send_text("b")
                        cv2.destroyAllWindows()

    def send_text(self, m):
        """ Sends text message to server (encoded)
        Params:
            m: string
                the message that will be sent to server
        """
        self.client_socket.send(m.encode())

    def receive(self, bts=300000):
        """ Used to receive message from server
        Params:
            bts: int
                represents how many bytes of data the client is ready to receive from the server
        Returns:
            pickled data or bytes
        """
        return self.client_socket.recv(bts)

    def receive_image(self, bts=300000, verbose=0):
        """ Receives pickled image from server and converts it to a numpy array
        Params:
            bts: int
                represents how many bytes of data the client is ready to receive from the server
            verbose: int or bool
                if True or 1, show details about the image, namely size and shape
        Returns:
            a numpy array representing the image
        """
        raw_data = self.receive(bts)  # receive response
        print("Received data is ", sys.getsizeof(raw_data), " bytes.")
        data = pickle.loads(raw_data, encoding='latin1')
        real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
        img = np.asarray(real_img)[:,:,::-1]
        if verbose:
            print("Received data is ", sys.getsizeof(raw_data), " bytes.")
            print("Image shape is ", img.shape)
            # Shape of the image: (height, width, colour channels)
        return  img




    def communicate(self):
        while True:

            server_action = self.client_socket.recv(2*1024).decode() # confirmation code
            start = time.time()
            if server_action == "send pepper":
                data = self.client_socket.recv(1000000)  # receive response
                print("Received data is ", sys.getsizeof(data), " bytes.")
                #print('Received from server: ' , data)  # show in terminal
                data = pickle.loads(data, encoding='latin1')
                real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                print("Received image shape = ", np.asarray(real_img).shape)
                #real_img = real_img.rotate(180)
                real_img.show()
                self.client_socket.send("d".encode())
                #print("type of data = ", type(data))
                # show numpy
            elif server_action == "pepper pred":
                data = self.client_socket.recv(2000000)  # receive response
                print("Received data is ", sys.getsizeof(data), " bytes.")
                #print('Received from server: ' , data)  # show in terminal
                data = pickle.loads(data, encoding='latin1')
                real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                img = np.asarray(real_img)[:,:,::-1]
                #break
                #print("data: ", img)
                #print("data type is ", type(img))
                #print("shape: ", img.shape)

                pred = self.dl_model.update(img)
                #print("pred type: ", type(pred))
                #print(pred)
                #break
                self.dl_model.draw(pred, np.ascontiguousarray(img), 0)
                #pred_dump = pickle.dumps(pred)
                #print("Sent prediction size = ", sys.getsizeof(pred_dump), " bytes.")
                self.client_socket.send("a".encode())

            elif server_action == "send image":
                data = self.client_socket.recv(2000000)  # receive response
                print("Received data is ", sys.getsizeof(data), " bytes.")
                data = pickle.loads(data, encoding="latin1")

                #img = Image.fromarray(data, "RGB")
                #img.show()
                self.client_socket.send("d".encode())
                #img = Image.frombytes("RGB", )
                # show dog image
                #plt.imshow(data)
                #plt.show()
                #print(type(data))

            elif server_action == "livestream":
                fps = pickle.loads(self.client_socket.recv(1000), encoding="latin1")
                duration = pickle.loads(self.client_socket.recv(1000), encoding="latin1")

                wait = 1/fps if fps is not None else 0
                if duration:
                    count = 0
                    start = time.time()
                    t_end = start + duration
                    frames = 0

                    # Visualisation ini

                    while time.time() < t_end:
                        data = self.client_socket.recv(300000)  # receive response
                        print("Received data is ", sys.getsizeof(data), " bytes.")
                        #print('Received from server: ' , data)  # show in terminal
                        data = pickle.loads(data, encoding='latin1')
                        real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                        img = np.asarray(real_img)[:,:,::-1]
                        print(img.shape)
                        #pred = self.dl_model.predict(img)

                        # Stream live data
                        cv2.imshow("stream", img)
                        cv2.waitKey(1)
                        self.client_socket.send("a".encode())
                        count += wait
                        frames += 1
                    cv2.destroyAllWindows()
                    end_time = time.time()-start
                    print("FPS test for input=80x60 resolution wireless:")
                    print("It took ", end_time, " seconds to stream ", frames, " frames.")
                    #print("count: ", count)
                    print("fps: ", fps)
                    #print("duration: ", duration)
                    print("true fps: ", frames/end_time)

                else:
                    while True:
                        try:
                            data = self.client_socket.recv(300000)  # receive response
                            print("Received data is ", sys.getsizeof(data), " bytes.")
                            #print('Received from server: ' , data)  # show in terminal
                            data = pickle.loads(data, encoding='latin1')
                            real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                            img = np.asarray(real_img)[:,:,::-1]
                            #pred = self.dl_model.predict(img)
                            pred = self.dl_model.update(img)
                            self.dl_model.draw(pred, np.ascontiguousarray(img), show=1,) #save_dir=os.path.join("pepper_test", "rotate"))
                            self.client_socket.send("a".encode())
                        except KeyboardInterrupt:
                            self.client_socket.send("b".encode())
                            cv2.destroyAllWindows()

            elif server_action == "bye":
                break

            print("It took ", time.time()-start, " seconds to process this request.")
        self.client_socket.close()  # close the connection
        #return real_img, pred

    def center_target(self, box, img_shape, stop_threshold = 0.1, vertical_offset=0.5, lost = None):
        """ Takes in target bounding box data and attempts to center it
        Preconditons:
            1. box must contain data about exactly 1 bounding box

        Params:
            box: 2D array
                Data about one bounding box in the shape of: 1 x 5
                Columns must be in the format of (x1, y1, x2, y2, id)
            img_shape: 1D array
                Shape of the original frame in the format: (height, width, colour channels),
                so passing in original_image.shape will do.
            stop_threshold: float between 0 and 1
                If the difference between box center and frame center over frame resolution is less than this threshold,
                tell the robot to stop rotating, otherwise, the robot will be told to rotate at a rate proportional to
                this ratio.
            vertical_offset: float between 0 and 1


        Returns:
            a string in the format a|b|c, where:
                a = message code, tells the server what mode it should be running (or what kind of data to expect)
                b = function the server will call
                c = parameter in the format of param_name=param param_name2=param2 param_name3=param3 ...
                        this part is also optional, because functions that b is referring to might not require params
        """
        if len(box)!=1:
            #raise Exception(f"The length of box is {len(box)}, but it should be 1!")
            #return "c$stop|"
            #return "c$stop|" + "$say|text=\"Target lost, searching for new target\"" if (len(box)==0 and self.dl_model.target_id is None) else "" + "$rotate_head_abs|"
            return "c$stop|" + "$target_lost|" if lost=="l" else "" + "$rotate_head_abs|"
            #return "c$stop|" + "$rotate_head_abs|"
        if len(img_shape)!=3:
            raise Exception(f"The shape of the image does not equal to 3!")

        # Since there's an extra dimension, we'll take the first element, which is just the single detection
        box = box[0]

        # Following shapes will be (x, y) format
        box_center = np.array([box[2]/2+box[0]/2, box[1]*(1-vertical_offset)+box[3]*vertical_offset])#box[1]/2+box[3]/2])
        frame_center = np.array((img_shape[1]/2, img_shape[0]/2))
        #diff = box_center - frame_center
        diff = frame_center - box_center
        horizontal_ratio = diff[0]/img_shape[1]
        vertical_ratio = diff[1]/img_shape[0]

        if abs(horizontal_ratio) > stop_threshold:
            #print("ratio = ", horizontal_ratio)
            # difference ratio greater than threshold, rotate at that ratio
            # locomotion_manager.walkToward(theta=horizontal_ratio)
            o = f"c$walkToward|theta={str(horizontal_ratio*0.9)}"
            #return f"c$walkTo|theta={str(horizontal_ratio*0.9)}"
        else:
            #return "c$stop|"
            o = self.approach_target(box, img_shape, command=f"rotate_head|forward={str(vertical_ratio*0.2)}")
            print("o = ", o)
        return o[0:2] + ("target_detected|$" if lost=="t" else "") +o[2:]

    def approach_target(self, box, img_shape, stop_threshold=0.65, move_back_threshold=0.8, command=""):
        # (x1, y1, x2, y2, id)
        box_area = (box[2]-box[0])*(box[3]-box[1])
        frame_area = img_shape[0]*img_shape[1]
        ratio = box_area/frame_area
        #return f"c|walkToward|x={str(1-ratio)}"
        if ratio > stop_threshold:
            if ratio > move_back_threshold:
                m = f"c$walkTo|x={str(ratio-1)}" # Move backwards
                #m = f"c$walkToward|x={str(ratio-1)}" # Move backwards
            else:
                m = "c$" + command
        else:
            #m = f"c$stop|$walkToward|x={str(1-ratio)}" # Move forward
            m = f"c$walkToward|x={str(1-ratio)}" # Move forward

            #m = f"c$walkTo|x={str(1-ratio)}" # Move forward
        #print("ratio = ", ratio)
        return m

if __name__ == '__main__':
    c = Client(image_size=[640,640])#, port=45093)
    c.neo_communicate()
    #c.communicate()