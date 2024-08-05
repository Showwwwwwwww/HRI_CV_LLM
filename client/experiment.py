from client3 import *
import  time

def communicateFlow_thread(device=2):
    c = Client()
    print("Client initizalized")
    c.communicate_behavior3()
    c.shutdown()

def head_tracking():
    c = Client()
    count = 0
    while True:
        print(f'count {count}')
        frame = c.get_image(save=True,path="./output",save_name="Pepper_Image",show=True)
        # print(f'image{count}')
        #cv2.imshow("Pepper_Image", img)
        #cv2.waitKey(1)
        # detecedPerson, track_id, face = c.face_recognition.process_frame(frame,
        #                                                                     show=False)  # Return the detected person
        box =  c.face_recognition.extractBox(frame)
        detecedPerson = 'SOme One '
        c.center_target2(detecedPerson,box, frame.shape)
        count+=1
    c.shutdown()

def audio_test():
    c = Client()
    c.rotate_head_abs()
    transcript = c.process_audio()
    if transcript is not None:
        c.say(word=transcript)
    else:
        c.say(word='No audio file generated')
    c.shutdown()

def testFPS():
    c = Client()
    num_frames = 100  # Get 100 images
    start_time = time.time()  # Start time

    for i in range(num_frames):
        c.process_image()

    elapsed_time = time.time() - start_time  # Total elapsed time
    fps = num_frames / elapsed_time  # Frames per second

    print(f"Captured {num_frames} frames in {elapsed_time:.2f} seconds. FPS: {fps:.2f}")
    
    c.shutdown()

def testImagesSpeed():
    c = Client()
    num_frames = 100  # Get 100 images
    start_time = time.time()  # Start time

    for i in range(num_frames):
        img = c.get_image(save=True, path="./output", save_name=f"Pepper_Image_{i}")

    elapsed_time = time.time() - start_time  # Total elapsed time
    fps = num_frames / elapsed_time  # Frames per second

    print(f"Captured {num_frames} frames in {elapsed_time:.2f} seconds. Images per second: {fps:.2f}")

    c.shutdown()
def checkSound():
    c = Client()
    x = 1
    while x < 100:
        sound = c.get_sound()
        print(sound)
        time.sleep(0.5)
        x+=1
    #c.rotate_head(forward=-0.7, left=0.7)
    time.sleep(5)
    # c.rotate_head(
    # time.sleep(5)
    c.shutdown()
def rotatHead():

    c = Client()
    c.rotate_head(forward=-0.7,left=0.7)
    time.sleep(5)
    # c.rotate_head(
    # time.sleep(5)
    c.shutdown()
if __name__ == '__main__':
    communicateFlow_thread()