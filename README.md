# Human Robot Interaction using Computer Vision and Large Language Models
This project develops a communication architecture enabling the Pepper robot to recognize users, remember them, and engage in meaningful interactions. It integrates various modules including tracking, face recognition, ASR (Automatic Speech Recognition), and LLM (Large Language Model).




## Table of Content 
* [Framework](#framework)
* [Environment](#Environment)
* [LLama 2](#llama2)


## Framework
This project utilizes a three-module architecture:
1. Client: Handles data processing and sends information to the server.
2. Server: Acts primarily as a communication relay between the client and Llama.
3. Llama: Processes requests and sends responses back to the robot.

Information exchange between the Client and Server is facilitated by a Flask server. Additionally, Llama operates its own server to simplify communications.

## Environment
### Client
#### Our Cuda & Python Version
- CUDA=11.8 
- Python=3.11.8

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
```
conda create --name server python=2.7
pip install -r ./server/requirements.txt
```
Export the naoqi Package is neccssary

```
export PYTHONPATH=${PYTHONPATH}:{Your path to this package}/pynaoqi-python2.7-2.5.5.5-linux64/lib/python2.7/site-packages
```

### Llama

```
conda create -n llama python=3.10.9
conda activate llama
conda install anaconda::cudatoolkit
```
#### File preparation for Llama Module
```
cd llama2
https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```
Then build the folder depeonds on your PC, please refer this file in [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md).

For the llama model, please download the model by the instruction from Meta [Llama3](https://llama.meta.com/) official website and place then into `llama.cpp/models`

## Usage
**Important: Start the server and llama before the client.**
### Server
```
conda activate server
```
```
cd server
python server.py
```

### Llama
```
conda activate llama
```
```
cd llama2/llama.cpp
CUDA_VISIBLE_DEVICES=0,1,2 ./llama-server \
    -m ./models/70B/Meta-Llama-3.1-70B-Instruct-Q3_K_M.gguf \
    --host "127.0.0.2" \
    --port 8080 \
    -c 4096 \
    -ngl 81 \
    --api-key "vl4ai"
```

### Client 
```
conda activate module
```
```
cd client
python experiment.py
```
