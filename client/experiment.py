from client2 import *

def communicateFlow(device=0):
    c = Client(device = device)
    c.communicate_behavior()
    c.shutdown()


def test_process_image():
    c = Client()
    count = 0
    while count < 100:
        img = c.get_image(save=True,path="./output",save_name="Pepper_Image")
        # print(f'image{count}')
        cv2.imshow("Pepper_Image", img)
        cv2.waitKey(1)
        c.process_image(image=img)
        count+=1
    c.shutdown()

def test_get_audio():
    c = Client()
    c.rotate_head_abs()
    x = c.get_audio()
    if x is not None:
        c.say(word='Generated audio file'   )
    else:
        c.say(word='No audio file generated')
    c.shutdown()

def test_process_audio():
    c = Client()
    c.process_audio(csv_path='./Whisper_speaker_diarization/output/transcript/transcript_result.csv',detected_person="Unknown")
    c.shutdown()

def test_say():
    c = Client()
    c.say(word='Hello World')
    c.shutdown()

def test_download_audio():
    c = Client()
    c.download_audio()
    c.shutdown()

if __name__ == '__main__':
    test_process_image()