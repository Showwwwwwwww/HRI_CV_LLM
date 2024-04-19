# Unlocking proactive behaviours in Pepper with deep learning

![pepper_bounding_robot](https://github.com/Wjjay/pepper_DL/assets/91184540/5e6d4534-fb58-45b0-9d35-c19ffeaac063)

Pepper, a versatile humanoid robot designed for human interaction, has gained significant popularity in various settings. However, it faces limitations in its default capabilities, including sensitivity to lighting conditions, limited human tracking, and the inability to adapt to modern computer vision techniques. To overcome these limitations and enhance Pepper's functionality, a framework is necessary. This framework integrates state-of-the-art deep learning (DL) models, enabling advanced perception and interaction with humans. By leveraging deep learning, Pepper can improve its visual perception, adapt to complex real-world scenarios, and deliver more satisfactory services in a wide range of environments. The development of this framework is crucial to enhance Pepper's sociability, acceptance, and effectiveness in different contexts.

## Table of Contents

* [Framework](#framework)
* [Default Follow Behaviour](#default-follow-behaviour)
* [DL Follow Behaviour](#dl-follow-behaviour)
* [Setup](#setup)
* [Execution](#execution)

## Framework

The framework requires 3 pieces of hardware:
1. Pepper
2. Another PC (ideally with a modern GPU, CPU, or both)
3. Wireless router

It solves Pepper’s compatibility issue with modern DL models by hosting two Python environments on a separate computer. The router establishes a local area network (LAN) that allows Pepper to connect to the server. 

The server environment, utilising Python 2 is responsible for communicating with Pepper’s operating system (OS) to:

1. extract live camera footage
2. execute commands

The client environment, powered by Python 3 can: 

1. control the robot through the server by sending Flask requests 
2. execute DL models with Pepper’s camera footage as input

Finally, the Pepper robot listens for commands from the server.

**Note that this framework only works for NAOqi 2.5**

## Default Follow Behaviour

Pepper’s default system, NAOqi 2.5, includes the ALPeoplePerception module, which uses Pepper’s front and 3D cameras to monitor surrounding humans. The default behaviour divides the area in front of Pepper into three engagement zones based on distance. Zone 1 is 0m to 1.5m away, Zone 2 is 1.5m to 2.5m away, and Zone 3 is beyond 2.5m. The maximum effective range of Pepper’s 3D camera, ASUS Xtion, is 3.5m, so it is reasonable to assume that Zone 3 ends there. Pepper’s default “follow” behaviour, developed using Choregraphe, waits for the keyword “follow me” as an audio cue to initiate tracking. It follows the person closest to it and maintains a default distance of 1m until the keyword “stop” is spoken 2. However, the speech recognition system often fails to register the keyword, resulting in poor user experience. It can be found [here](https://github.com/Wjjay/pepper_DL/tree/pepper_flask/FollowCome2Me)

To consistently trigger Pepper's tracking mode, we changed this behaviour to be triggered by its haptic sensors located on its head rather than through voice command. This modified version is available [here](https://github.com/Wjjay/pepper_DL/tree/pepper_flask/FollowMeTap).

## DL Follow Behaviour

The DL-based follow behaviour spins around to search for potential targets using [YOLO-Pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose), which allows it to capture both keypoint and bounding box data of detected individuals. If a potential target raises their hand, the system can tell by calculating the difference between the wrist and shoulder keypoint. Then, when the system is sure that a potential target is intentionally raising their hand, it'll filter out every other detection and engage in tracking mode, where the filtered data is passed onto of the trackers [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), or [OC-SORT](https://github.com/noahcao/OC_SORT). We'll get a tracking ID from the tracker, and for every subsequent tracker output, the robot will only focus on the target with the same ID that we just saved.

A high level diagram depicts what happens during run-time:

![uml-sequence](https://github.com/Wjjay/pepper_DL/blob/pepper_flask/assets/imgs/dl_behaviour.png)

## Setup

To use this code, you need to set up the hardware and software. Let's start with the hardware:
1. With the router, set up a LAN
2. Connect the PC and Pepper to the LAN
3. On Pepper's tablet, search for the version of its OS and make sure it's 2.5.x
4. On Pepper's tablet, in the Wi-Fi tab, find its IP address and record it somewhere

For the software, follow these steps:
1. Clone [this repo](https://github.com/Wjjay/pepper_DL/tree/pepper_flask) and make sure the branch is **pepper_skeleton**
2. On Anaconda, create a Python 2, and a Python 3 virtual environment
3. Visit this [website](https://www.aldebaran.com/en/support/pepper-naoqi-2-9/downloads-softwares) and download the Python 2.7 SDK with the same OS that you're using, and the same NAOqi version as your Pepper robot[^1]
4. Unzip the Python SDK and copy-paste its contents into the Python 2 environment that you've just created, replacing everything
5. Activate the Python 2 environment, and install the packages in [requirements.txt](https://github.com/Wjjay/pepper_DL/blob/pepper_flask/server/requirements.txt) [^2]
6. Activate the Python 3 environment, and install the packages in [requirements1.txt](https://github.com/Wjjay/pepper_DL/blob/pepper_flask/client/requirements1.txt) 
7. In the same Python 3 environment, install the packages in [requirements2.txt](https://github.com/Wjjay/pepper_DL/blob/pepper_flask/client/requirements2.txt) [^3]

[^1]: *This has been tested on Linux (Ubuntu) and it works. We haven't tested it on other OS, but there should be no reason for them to not work.*
[^2]: *Pip may not necessarily install all dependencies for you. So, when you run the program later, it may tell you to install additional dependencies*
[^3]: *In our project, we used Pytorch, but because there are many ways to install it depending on your hardware, you have to figure this out yourself*

## Execution

Before execution, make sure that Pepper's surrounding is clear, otherwise, you may risk damaging the robot.

To run the behaviour:
1. Turn on Pepper, and double-tap its chest button to make it turn of automatic life. It'll assume this lifeless posture if you're successful
2. Open up two terminals, and activate the Python 2 and Python 3 environment with them
3. In the Python 2 terminal, cd into the directory of [server](https://github.com/Wjjay/pepper_DL/tree/pepper_flask/server)
4. In the Python 3 terminal, cd into the directory of [client](https://github.com/Wjjay/pepper_DL/tree/pepper_flask/client)
5. In the Python 2 terminal, run `python server.py --ip *Pepper's IP address*`, if successful, Pepper should stand up-right again
6. In the Python 3 terminal, run `python experiments.py` (should run the follow behaviour with OC-SORT), if successful, Pepper should start rotating to look for potential targets. At this stage, you can raise your hand to initiate its tracking mode
