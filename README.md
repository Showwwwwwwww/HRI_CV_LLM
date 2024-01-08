# robot_research
This repo store all the information relats to  Honor Research



## Llama 2 setting
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