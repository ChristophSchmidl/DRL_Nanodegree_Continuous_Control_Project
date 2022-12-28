# Udacity Nanodegree - Deep Reinforcement Learning

<img src="img/reacher.gif" width="650">

## Project 2: Continuous Control

The second project in this Nanodegree is about continuous control using the Unity ML-Agents Reacher Environment. The environment provides double-jointed arms that should move to certain target locations. These target locations are represented as moving spheres. The end-piece of such an arm should maintain its moving position inside a target sphere for as long as possible.

### Project Details

A **reward** of **+0.1** is provided for each step that the agent's hand is in the goal location. Therefore, the goal is to maintain its position at the target location for as many time steps as possible. This also means that there are no negative rewards (penalties).

The **observation/state space** consists of **33** dimensions corresponding to position, rotation, velocity, and angular velocities of the arm. 

Each **action** is a **4-dimensional vector**, corresponding to torque applicable to two joints. Every entry in the action vector should be a number (continuous? float?) between -1 and 1.

There are two separate versions of the environment

- with a **single agent**
- with **20 identical agents**, each with its own copy of the environment

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

You only need to solve **one** of the two environment options.

#### Option 1: Solve the single agent environment

The task is episodic, and in order to solve the environment, your agent must get an average score of **+30 over 100 consecutive episodes**.

#### Option 2: Solve the multi agent (duplicates) environment 

The agents must get an average score of **+30** (over 100 consecutive episodes, and over all agents)

- After each episode, the rewards that each agent received are summed (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. The average of these 20 scores is then computed.
- This yields an **average score** for each episode (where the average is over all 20 agents).


### Getting Started

This repository was implemented with Python version 3.9.13. The following steps should enable you to reproduce and test the implementation.

- Create a python virtual environment at the root: ``python -m venv venv``
- Activate the virtual environment: ``source venv/bin/activate`` (if you are using Linux/Ubuntu)
- Upgrade pip: ``pip install --upgrade pip`` (optional)
- Install dependencies from local folder: ``pip install ./python``

After these instructions, everything should be ready to go. However, if you encounter compatibility issues with your CUDA version and Pytorch, then you could try to solve these problems by installing a specific PyTorch version that fits your CUDA version. In my case, I could resolve it using the following commands:

- ``pip uninstall torch``
- ``pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html``

The repository already contains the unity environments for Linux under the following locations:

-  ``src/Reacher_Linux_{one_agent|one_agent_NoVis|twenty_agents|twenty_agents_NoVis}``

However, if you want to install the unity environments for a different operating systems then you can find the download instructions below.


#### Download the Unity Environment

**Version 1: One (1) Agent**

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip) (**Note**: You can replace ``Reacher_Linux.zip`` at the end of the URL with ``Reacher_Linux_NoVis.zip`` to get the non-visual environment. I already included both in the repo.)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**Version 2: Twenty (20) Agents**

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) (**Note**: You can replace ``Reacher_Linux.zip`` at the end of the URL with ``Reacher_Linux_NoVis.zip`` to get the non-visual environment. I already included both in the repo.)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### Instructions

After cloning the repository and installing all necessary dependencies, you can train and evaluate the different agents through the command line. The file ``src/main.py`` is using the ``argparse`` library to parse the command line arguments. You can use the following command to see all available arguments:

``python -m src.main --help``

which outputs the following:

```
ActorCritic methods - Continuous Control Project

optional arguments:
  -h, --help            show this help message and exit
  -gpu GPU              GPU: 0 or 1. Default is 0.
  -episodes EPISODES    Number of games/episodes to play. Default is 1000.
  -alpha ALPHA          Learning rate alpha for the actor network. Default is 0.0001.
  -beta BETA            Learning rate beta for the critic network. Default is 0.0001.
  -gamma GAMMA          Discount factor for update equation
  -tau TAU              Update network parameters. Default is 0.001.
  -algo ALGO            You can use the following algorithms: DDPGAgent. Default is DDPGAgent.
  -buffer_size BUFFER_SIZE
                        Maximum size of memory/replay buffer. Default is 1000000.
  -batch_size BATCH_SIZE
                        Batch size for training. Default is 128.
  -load_checkpoint      Load model checkpoint/weights. Default is False.
  -model_path MODEL_PATH
                        Path for model saving/loading. Default is data/
  -plot_path PLOT_PATH  Path for saving plots. Default is plots/
  -use_eval_mode        Evaluate the agent. Deterministic behavior. Default is False.
  -use_multiagent_env   Using the multi agent environment version. Default is False.
  -use_visual_env       Using the visual environment. Default is False.
  -save_plot            Save plot of eval or/and training phase. Default is False.
```

#### Training

If you want to start training the agents from scratch with default hyperparamters, you can use the following command:

- ``python -m src.main -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES>``

Note: At the moment only agent "DDPGAgent" is available.

#### Evaluation

If you want to evaluate the trained agents in non-visual mode (fast), you can use the following command:

- ``python -m src.main -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES> -use_eval_mode`

The above command simply loads the appropriate model weights, sets the noise to 0.0 to enforce a deterministic behavior (no exploration, pure exploitation) and runs the agent in non-visual mode.

If you want to see the trained agents in action in visual mode (slow), you can use the following command:

- ``python -m src.main -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES> -use_eval_mode -use_visual_env``


### Report

If you are interested in the results and a more detailed report, please have a look at [report](REPORT.md).