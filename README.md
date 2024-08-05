# Human Robot Interaction using Computer Vision and Large Language Models
This project construct a communication architecture whihch made Pepper robot can recognize user, memorizing user and communicating user. Which involving the module of tracking, face recognition, ASR and LLM. 

## Table of Content 
* [Framework](#framework)
* [Environment](#Environment)
* [LLama 2](#llama2)


## Framework
The framework required 3 modules to running:
1. Client 
2. Server
3. Llama

In this project, we treat the robot as the rbot who will only send the receive information as server, these information will be send to the client for processing, meanwhile, the client will send the information to llama to get the response to robot to make it communicate with user. 

## Environment
### Client
#### Testing Cuda & Python Version
CUDA=11.8 Python=3.11.8

#### Install requirements file
```angular2html
conda create -n module -c conda-forge cudatoolkit=11.8 cudnn cudatoolkit-dev torchvision python=3.11.8 
conda activate module
pip install -r ./client/requirements.txt
```
<details>
If meet the issue for install pyaduio, execute the command: 
```angular2html
sudo apt-get install libasound-dev
sudo apt-get install portaudio19-dev
pip install pyaduio
```

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
</details>

### Server
conda create 

### Llama

```
conda create -n llama python=3.10.9
conda activate llama
conda install anaconda::cudatoolkit
```
For the llama model, please download the model by the instruction from Meta [Llama3](https://llama.meta.com/) official website


## Usage

## Naoqi Export
```
export PYTHONPATH=${PYTHONPATH}:/home/shuo/robot_research/pynaoqi-python2.7-2.5.5.5-linux64/lib/python2.7/site-packages
```