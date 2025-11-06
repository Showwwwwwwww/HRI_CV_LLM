# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Human-Robot Interaction (HRI) system that integrates Pepper robot with advanced AI capabilities:
- **Whisper**: Automatic speech recognition with speaker diarization
- **Llama2**: Large language model for conversation via llama.cpp
- **YOLOv8**: Object and person detection
- **InsightFace**: Face recognition and analysis

## Architecture

The system uses a **client-server architecture**:

### Server (`server/`)
- Flask REST API server running **on the Pepper robot** (Python 2.7, NAOqi framework)
- Provides endpoints for:
  - `/image/send_image`: Get camera feed from Pepper
  - `/audio/recording`: Record audio from Pepper's microphones
  - `/audio/volume`: Get current audio volume level
  - `/voice/say`: Make Pepper speak (TTS)
  - `/locomotion/rotateHead`: Control Pepper's head movement
- Requires connection to Pepper robot at IP (default: 192.168.0.52, port: 9559)
- Modules: `camera.py`, `audio2.py`, `voice.py`, `head.py`

### Client (`client/`)
- Main control logic running on **external machine with GPU**
- `client2.py`: Core `Client` class that orchestrates all modules
- `llm_agent.py`: LLM-only interaction mode
- `llm_vision_agent.py`: LLM + vision interaction mode
- Communicates with Pepper via HTTP requests to server endpoints

### Llama2 (`llama2/`)
- Contains llama.cpp implementation with quantized Llama2 models
- Models stored in: `llama-2-7b-chat/` and `llama-2-13b-chat/`
- Compiled C++ binary in `llama.cpp/main` for inference

## Environment Setup

This project requires **two separate conda environments** due to incompatible dependencies:

### Environment 1: `llama` (for Llama2)
```bash
conda create -n llama python=3.10.9
conda activate llama
conda install anaconda::cudatoolkit
```
Note: Llama2 conversion scripts require numpy==1.24

### Environment 2: `module` (for Whisper & Vision)
```bash
conda create -n module -c conda-forge cudatoolkit=11.8 cudnn cudatoolkit-dev torchvision python=3.11.8
conda activate module
pip install -r ./client/Whisper_speaker_diarization/requirements.txt
pip install -r ./client/Visual/requirements.txt
```

If PyAudio installation fails:
```bash
sudo apt-get install libasound-dev portaudio19-dev
pip install pyaudio
```

### NAOqi SDK
For Pepper robot integration, add to your environment:
```bash
export PYTHONPATH=${PYTHONPATH}:/path/to/pynaoqi-python2.7-2.5.5.5-linux64/lib/python2.7/site-packages
```

## Building Llama.cpp

### Linux with GPU (CUDA)
Navigate to `llama2/llama.cpp/` and:
1. Check GPU architecture: `nvcc --list-gpu-arch`
2. Edit `Makefile` line 245: change `native` to your architecture (e.g., `sm_87` for RTX 4090)
3. Build with CUDA:
```bash
make LLAMA_CUBLAS=1
```

### macOS
CPU only:
```bash
make LLAMA_NO_METAL=1
```

GPU (Apple Silicon):
```bash
LLAMA_METAL=1 make
```

## Running the System

### 1. Start Pepper Server
On the Pepper robot (or machine connected to Pepper):
```bash
cd server/
python server.py --ip 192.168.0.52 --port 9559
```

### 2. Run Client (LLM-only mode)
```bash
cd client/
conda activate module
python llm_agent.py --device 1  # device = CUDA device number
```

### 3. Run Client (LLM + Vision mode)
```bash
cd client/
conda activate module
python llm_vision_agent.py  # Uses device 1 by default
```

### 4. Run Llama2 Directly (for testing)
```bash
cd llama2/llama.cpp/
./main -m ./models/7B/ggml-model-q4_0.bin -n 4096 --repeat_penalty 1.0 --color -i -r "User:" -f ./prompts/customisedChatPrompt.txt
```

Set specific GPU:
```bash
CUDA_VISIBLE_DEVICES=1 ./main -m ./models/7B/ggml-model-q4_0.bin ...
```

## Key Components

### Whisper Speaker Diarization (`client/Whisper_speaker_diarization/`)
- Main class: `Whisper` in `whisper.py`
- Performs speech-to-text with speaker identification
- Uses models: "medium" (default) or "large-v2" for better accuracy
- Hugging Face demo: https://huggingface.co/spaces/vumichien/Whisper_speaker_diarization

### Face Recognition (`client/Visual/`)
- Main class: `FaceRecognition2` in `detection2.py`
- Uses InsightFace (buffalo_l model) for face embedding
- Face database stored in `./database/face_db/`
- File naming convention: `name_age_gender.jpg`
- Returns detected person names and generates conversation prompts

### Client Control Flow (`client/client2.py`)
The `Client` class main workflow:
1. `get_image()`: Request image from Pepper server
2. `process_image()`: Detect faces and generate visual context
3. `sound_exceed_threshold()`: Check if audio input is present
4. `process_audio()`: Record and transcribe speech with speaker diarization
5. Send transcript to Llama2 for response generation
6. `say()`: Send response to Pepper for speech output

## Important Notes

- **llmControl module**: The import `from llmControl import llm` in `client2.py` suggests there's a missing module that interfaces with Llama2. You may need to implement this or it may be in a private/local path.
- **YOLOv8 model**: `yolov8s.pt` is included in the repository root and `client/` directory
- **Output directory**: Results are logged to `./output/` as JSON files
- **Python 2.7**: Server code runs on Python 2.7 due to NAOqi SDK requirements (Pepper robot)
- **Python 3.11**: Client code requires Python 3.11 for Whisper and vision modules

## Development Tips

- Use `CUDA_VISIBLE_DEVICES` environment variable to control which GPU is used
- Enable CUDA debugging with: `PYTORCH_USE_CUDA_DSA=1` and `CUDA_LAUNCH_BLOCKING=1`
- The system logs conversation data to JSON files in `./output/`
- Pepper's default server address is `http://localhost:5001` when running locally
