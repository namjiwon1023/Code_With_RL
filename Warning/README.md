# Warning (This is a test version)

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/namjiwon1023/Code_With_RL
    cd Code_With_RL
    ```
- If you don't have Pytorch installed already, install your favourite flavor of Pytorch. In most cases, you may use
    ```bash
    pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html # pytorch 1.8.1 LTS CUDA 10.2 version. if you have GPU.
    ```
    or
    ```bash
    pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html # pytorch 1.8.1 LTS CPU version. if you don`t have GPU.
    ```
    to install Pytorch GPU or CPU version.

## File Structure
+ **Hyperparameter** # Algorithm Hyperparameters
  + dqn.yaml
  + doubledqn.yaml
  + duelingdqn.yaml
  + d3qn.yaml
  + noisydqn.yaml
  + ddpg.yaml
  + td3.yaml
  + sac.yaml
  + ppo.yaml
  + a2c.yaml
  + behaviorcloning.yaml
  + etc.
+ **agent.py**
  + reinforcement learning algorithm
+ **network.py**
  + QNetwork
  + NoisyLinear
  + ActorNetwork
  + CriticNetwork
+ **replaybuffer.py**
  + Simple PPO Rollout Buffer
  + Off-Policy Experience Replay
+ **runner.py**
  + Training loop
  + Evaluator
+ **main.py**
  + Start training
  + Start evaluation
+ **utils.py**
  + Make gif image
  + Drawing
  + Basic tools

## Quick Start

To **train** a new network : run `python main.py --algorithm=selection algorithm`
To **test** a preTrained network : run `python main.py --algorithm=selection algorithm --evaluate=True`

Reinforcement learning **algorithms** that can now be selected:
+ **DQN**
+ **Double_DQN**
+ **Dueling_DQN**
+ **D3QN**
+ **Noisy_DQN**
+ **DDPG**
+ **TD3**
+ **SAC**
+ **PPO**
+ **A2C**
+ **BC_SAC**

## Requirements

```
Python 3.6+
Pytorch 1.6+ : https://pytorch.org/get-started/locally/
Numpy
openai gym : https://github.com/openai/gym
matplotlib
tensorboard

```