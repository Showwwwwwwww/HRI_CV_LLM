# Human Robot Interaction using Computer Vision and Large Language Models
This project demonstrates the integration of advanced technologies such as Llama2, Whisper, and Yolov8 into robotic systems to significantly improve Human-Robot Interaction (HRI) performance.

## Table of Content 
* [Framework](#framework)
* [Whisper Diriazation](#whisper)
* [LLama 2](#llama2)


## Framework
The framework required 4 pieces of module to running:
1. Whisper Diaziration
2. Llama2
3. Pepper
4. Yolov8


## Whisper 
Whisper is the ASR techniques we used in this project, meanhwhile, for the diaziration version, it can recognize different speakers in one audio and assign people for each sentence. 

Try this module online by following this link: https://huggingface.co/spaces/vumichien/Whisper_speaker_diarization 

### Set up

1. Clone [This repo](https://github.com/Showwwwwwwww/HRI_CV_LLM.git)
```
git clone https://github.com/Showwwwwwwww/HRI_CV_LLM.git
```

2. Create a virtual environment with python version == 3.11

3. Navigate to Whisper_speaker_diarization folder and install the package and requirement file

```
brew install ffmpeg
```

```
pip install -r requirements.txt
```

4. Run it by the command
```
python app.py
```


## Llama2
### Different environment for Llama 2
"Precondition" -  The file who transfer the code in Llama to cpp format required  dnumpy==1.24, which is different to the requirement for Whisper_speaker_diarization.  
Follwing this link: https://www.youtube.com/watch?v=TsVZJbnnaSs 

./main -m ./models/7B/ggml-model-q4_0.bin -n 1024 --repeat_penalty 1.0 --color -i -r "User:" -f ./prompts/chat-with-bob.txt

First you have to compile your program. For a cpp program usually g++ is used. So compile it with

g++ -Wall -o prg prg.cpp
Afterward you have to modify your access with

chmod +x prg
to be able to invoke the program.

Now you can call your program with your arguments:

./prg arg1 arg2 arg3