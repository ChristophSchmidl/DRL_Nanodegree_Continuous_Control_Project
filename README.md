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



#### Training



#### Evaluation




### Report

If you are interested in the results and a more detailed report, please have a look at [report](REPORT.md).