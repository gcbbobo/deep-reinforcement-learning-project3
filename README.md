# deep-reinforcement-learning-project3
###########################################################################################

All the links below are listed without asking for permission. Any owner of the resources/ author of the articles
can contact me through email to remove the links in README.md that connect to their online properties. 

My email address is: freedomgu@foxmail.com. Apologize for any inconvenience in advance. 

###########################################################################################
## 1. The Environment
Two agents controlling rackets are trying to bounce a ball over a net and keep the ball in play.
![image](https://github.com/gcbbobo/deep-reinforcement-learning-project3/blob/master/competition.PNG)

#### Reward:
An agent hits the ball over the net:                                 reward +0.10 for this agent

An agent lets a ball hit the ground or hits the ball out of bounds:  reward -0.01 for this agent

#### State:
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 

Each agent receives its own, local observation.

#### Action:
2 continuous actions available for each agent:

movement toward (or away from) the net, and jumping.

#### Target:
An average score of +0.5 over 100 consecutive episodes, after taking the maximum over both agents. 

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

#### In folder: update_every_multiple_steps

Tennis_TAU_1e-1_update_30.ipynb + ddpg_agent.py + model.py:

Train the more intelligent agent to achieve mean score >= 0.5 with 2 actor and 1 critic

The network weights are updated every 30 time steps.

#### In folder: min_as_settings

Tennis_min.ipynb + ddpg_agent.py + model.py:

Train the less intelligent agent to achieve mean score >= 0.5 with 2 actors and 1 critic

#### In folder: copy_param_ref

Tennis_v6.10_original.ipynb + ddpg_agent.py + model.py:

Train the more intelligent agent to achieve mean score >= 0.5 with 2 actors and 1 critic

New comments, new parameters.

The training speed is improved rapidly and the problem is solved in only several hundred episodes.


#### In folder: paper_version_not_solved

Tennis_ref_to_paper(not_learning).ipynb + ddpg_agent.py + model.py:

Try to train the more intelligent agent to achieve mean score >= 0.5 with 2 actors and 2 critics

The actors only receives local information and the critics use global information during training except rewards.

The idea follows that put forward in the paper.

This version is incomplete and the agent is not learning at present.

Parameters need further tuning.


