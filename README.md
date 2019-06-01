# deep-reinforcement-learning-project3
###########################################################################################

All the links below are listed without asking for permission. Any owner of the resources/ author of the articles
can contact me through email to remove the links in README.md that connect to their online properties. 

My email address is: freedomgu@foxmail.com. Apologize for any inconvenience in advance. 

###########################################################################################
## 1. The Environment
The physical model of the problem is a double-jointed arm which is designed to move to target locations. The states are the 33 physical states of the system, and the control inputs are torque applicable to two joints. The figure below shows a multi-agent case. To simplify the problem, only the single agent case is considered. 

![image](https://github.com/gcbbobo/deep-reinforcement-learning-project2/blob/master/reacher.PNG)

#### Reward:
Agent's hand in goal location: reward +0.1/each step

#### State:
The state space has 33 dimensions including position, rotation, velocity, and angular velocities of the two arm Rigidbodies.

#### Action:
4 continuous actions available:

torques applicable to two joints of the agent, ranging from -1 to 1.

#### Target:
An average score of +30 over 100 consecutive episodes.

Details can be found:

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md

Some interesting difference lies there. As defined in the introduction above, the state space should have 26 dimensions. However, the agent I trained is 33 dimensional. 

## 2. Installation
#### Anaconda Installation and Virtual Environment Setup
Follow the instructions listed: 

https://inmachineswetrust.com/posts/deep-learning-setup/

Be sure to work with python 3.x. As far as I can remember, I particularly downloaded python 3.6.8 to work with ML-Agents and I am not sure which versions would also work well.

#### CUDA and cuDNN
If GPU available, go to the official website for CUDA and cuDNN.Download packages that corresponding to the GPU. 

For GPU Training using The ML-Agents Toolkit, only CUDA v9.0 and cuDNN v7.0.5 is supported at present.

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md

#### TensorFlow
For ml agent:

`pip install tensorflow==1.7.1`

`pip install tensorflow-gpu==1.7.1`

#### PyTorch
Follow the official guide:

https://pytorch.org/get-started/locally/

Download the CPU version.

Some commands taken by myself:

`conda install pytorch-cpu torchvision-cpu -c pytorch`

`pip install --user https://download.pytorch.org/whl/cu100/torch-1.0.1-cp36-cp36m-win_amd64.whl`

`pip install --user torchvision`

My PyTorch version for CPU is 1.0.1.

#### Box2d
`conda install swig` # needed to build Box2D in the pip install

`pip install box2d-py` # a repackaged version of pybox2d

#### pip3
Use conda and cd to the dictionary of any (virtual) environment.

In my case, 2 virtual environments are set up in addition to the root environment. Their locations are:

C:\download\anaconda\envs\dl

C:\download\anaconda\envs\tensorflow2

C:\download\anaconda

Take the first environment as an example.

Open Anaconda Prompt and enter:

`cd C:\download\anaconda\envs\dl`

Then type:

`python -m pip install --upgrade pip --force-reinstall`

Repeat this process for each environment.

#### Unity
https://unity.com/

The version of Unity should be compatible with Python or something interesting may show up like:

mlagents.envs.exception.UnityEnvironmentException: 

The API number is not compatible between Unity and python. Python API : API-7, Unity API : API-4

Someone report similar problems on github for other reasons:

https://github.com/Unity-Technologies/ml-agents/issues/1770

One feasible solution seems to be using "pip3" command instead of "pip" during installation of packages.

Personally, I downloaded Unity 2018.2.8f1 and it works well with all the other packages. 

#### ML-Agent
Thank to Rahul's kind assistance, I just follow instructions provided by:

https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Unity-ml-agents-on-Windows

Download all the rest packages with "pip3" command.

ML-Agent can be downloaded from the official website.

A link for my ML-Agent package is provided:

https://drive.google.com/file/d/1eguoN8lslH5qmRKJoBGKoyyVSSnH835E/view?usp=sharing

## 3.Instructions
The following codes are developed based on the sample codes provided by Udacity.


