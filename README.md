# Code With Reinforcement Learning (Under Construction)

## Single Agent Algorithm

### Value Based

- [x] Deep Q Network(DQN) (off-policy)
- [x] Double Deep Q Network(Double DQN) (off-policy)
- [x] Dueling Deep Q Network(Dueling DQN) (off-policy)
- [x] Duelling Double Deep Q Network(D3QN) (off-policy)
- [x] Noisy Networks for Exploration(NoisyDQN) (off-policy)

### Actor-Critic Method

- [ ] Advantage Actor-Critic(A2C) (on-policy)
- [ ] Asynchronous Advantage Actor-Critic(A3C) (on-policy)
- [x] Proximal Policy Optimization(PPO)(GAE) (on-policy)(Nearing off-policy)
- [ ] Proximal Policy Gradient(PPG) (on-policy PPO + off-policy Critic[Let it share parameters with PPO's Critic])
- [x] Deep Deterministic Policy Gradient(DDPG) (off-policy)
- [x] Twin Delayed Deep Deterministic policy gradient(TD3) (off-policy)
- [x] Soft Actor-Critic(SAC) (off-policy)

### Imitation Learning / Inverse Reinforcement Learning

- [x] Behavior Cloning(BC)
- [ ] Generative Adversarial Imitation Learning(GAIL)

### ReplayBuffer Structure

- [ ] Prioritized Experience Replay(PER)
- [ ] Hindsight Experience Replay(HER)


## Multi Agent Algorithm

### Actor-Critic Method

- [x] Multi Agent Deep Deterministic Policy Gradient(MADDPG)
- [x] MADDPG Method TD3, SAC
- [ ] Multi Agent Proximal Policy Optimization(MAPPO)
- [ ] COMA

### Value Based

- [ ] QMIX


## Quick Start

Simply run:

`python main.py`

for default args. Changeable args are(For detailed training hyperparameters, please view `utils/arguments.py`):
```
--env-name: String of environment name (Default: Pendulum-v0)
--seed: Int of seed (Default: 123)
--render： Whether the training environment is visualized (Default: False)
--time-steps: number of time steps (Default: 3000000)
--episode: number of episode (Default: int(1e6))
--actor-lr: actor network learning rate (Default: 1e-4)
--critic-lr: critic network learning rate (Default: 1e-3)
--hidden-size: hidden layer units (Default: 128, According to the algorithm, according to the environment, you need to make changes by yourself)
--target_update_interval： Target network update frequency (Default: 1)
--epsilon: Random exploration probability (Default: 1.0)
--min_epsilon： Randomly explore the minimum probability (Default: 0.1)
--epsilon_decay: Attenuation rate of epsilon (Default: 0.0001)
--gamma: discount factor (Default: 0.99， According to the algorithm, according to the environment, you need to make changes by yourself)
--tau: parameter for updating the target network (Default: 5e-3， According to the algorithm, according to the environment, you need to make changes by yourself）
--buffer-size： number of transitions can be stored in buffer (Default: int(1e6)）
--batch-size: number of episodes to optimize at the same time (Default: 256， According to the algorithm, according to the environment, you need to make changes by yourself)
--save-dir: directory in which training state and model should be saved (Default: "./model")
--evaluate-episodes: number of episodes for evaluating (Default: 10)
--evaluate: whether to evaluate the model (Default: False)
--evaluate-rate: how often to evaluate model (Default: 1000)
```