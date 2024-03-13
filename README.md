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
3. Pepper(Incomplete)
4. Yolov8(Incomplete)


## Whisper 
Whisper is the ASR techniques we used in this project, meanhwhile, for the diaziration version, it can recognize different speakers in one audio and assign people for each sentence. 

Try this module online by following this Huggingface link: https://huggingface.co/spaces/vumichien/Whisper_speaker_diarization 

### Setup

1. Clone [This repo](https://github.com/Showwwwwwwww/HRI_CV_LLM.git)
```
git clone https://github.com/Showwwwwwwww/HRI_CV_LLM.git
```

2. Create a virtual environment with python version == 3.11

3. Navigate to Whisper_speaker_diarization folder and install the package(ffmpeg) and requirement file by following command

```
brew install ffmpeg
pip install pynput
pip install PyAudio
pip install sounddevice
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

### Installation
The setup for Llama2 need to request for download the llama2 model from [facebook](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmxkY2pESDMxOWNqVHBlTU1TMVAtOVFpeVVnZ3xBQ3Jtc0tuT1RsX2ZQZFBjWEZlRDA4QWVUbFhvNzNQbDg3ejBuRzFoSTJCM1Jzcm4xM2pLVHBXRHQtaWJIRVJXNW1HLUw4NG9WQW5xTk9LWVR0aUJzNzlYRzdhakNldl9jREdVc1gxZjU0WGZuclhNSWlDRkdURQ&q=https%3A%2F%2Fai.meta.com%2Fresources%2Fmodels-and-libraries%2Fllama-downloads%2F&v=TsVZJbnnaSs) at first, and request code for [this repo](https://github.com/facebookresearch/llama) to get the model for llama2. Then quantize the model in the [llama.cpp](https://github.com/facebookresearch/llama) in the [llama2 folder](https://github.com/Showwwwwwwww/HRI_CV_LLM/tree/main/llama2/llama.cpp) in this repo

The complete process is Follwing this youtube video from Alex Ziskind : https://www.youtube.com/watch?v=TsVZJbnnaSs 

### Usage
Please follow processes in [doc](https://github.com/Showwwwwwwww/HRI_CV_LLM/tree/main/llama2/llama.cpp#usage)

### Running
First you have to compile your program. For a cpp program usually g++ is used. So compile it with
```
g++ -Wall -o prg prg.cpp
```
Afterward you have to modify your access with
```
chmod +x prg
```
to be able to invoke the program.

Now you can call your program with your arguments:

./prg arg1 arg2 arg3

Example:
```
./main -m ./models/7B/ggml-model-q4_0.bin -n 1024 --repeat_penalty 1.0 --color -i -r "User:" -f ./prompts/chat-with-bob.txt
```