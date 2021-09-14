import torch as T
import numpy as np
import random

mbpo_target_entropy_dict = {'Hopper-v2':-1, 'HalfCheetah-v2':-3, 'Walker2d-v2':-3, 'Ant-v2':-4, 'Humanoid-v2':-2}
mbpo_epoches = {'Hopper-v2':125, 'Walker2d-v2':300, 'Ant-v2':300, 'HalfCheetah-v2':400, 'Humanoid-v2':300}


def _evaluate_agent(env, agent, args, n_starts=10, render=False):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            if render:
                env.render()
            action = agent.select_test_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = next_state
    return reward_sum / n_starts

def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False

def network_update(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def _random_seed(env, test_env, seed):
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    test_env.seed(seed+9999)
    test_env.action_space.np_random.seed(seed+9999)

def get_batch_buffer(buffer, batch_size, device, n_actions):
    with T.no_grad():
        samples = buffer.sample_batch(batch_size)
        state = T.as_tensor(samples['state'], dtype=T.float32, device=device)
        next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=device)
        action = T.as_tensor(samples['action'], dtype=T.float32, device=device).reshape(-1, n_actions)
        reward = T.as_tensor(samples['reward'], dtype=T.float32, device=device).reshape(-1, 1)
        mask = T.as_tensor(samples['mask'], dtype=T.float32, device=device).reshape(-1, 1)
    return state, next_state, action, reward, mask

# model save functions
def _save_model(net, dirpath):
    T.save(net.state_dict(), dirpath)

# model load functions
def _load_model(net, dirpath):
    net.load_state_dict(T.load(dirpath))

def _target_soft_update(target_net, eval_net, tau):
    for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
        t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins
