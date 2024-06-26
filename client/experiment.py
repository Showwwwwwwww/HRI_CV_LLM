from client2 import *

def communicateFlow(device=0):
    c = Client(device = device)
    c.communicate_behavior()
    c.shutdown()

def communicateFlow_thread(device=1):
    c = Client(device = device)
    c.communicate_behavior2()
    c.shutdown()

def head_tracking():
    c = Client(device = 0)
    count = 0
    while count < 100:
        img = c.get_image(save=True,path="./output",save_name="Pepper_Image")
        # print(f'image{count}')
        #cv2.imshow("Pepper_Image", img)
        #cv2.waitKey(1)
        c.process_image(image=img)
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

if __name__ == '__main__':
    communicateFlow_thread()