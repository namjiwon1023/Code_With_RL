import torch as T
import numpy as np

def _evaluate_agent(env, agent, args, n_starts=10, render=False, evaluate=True):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            if render:
                env.render()
            action = agent.choose_action(state, evaluate)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = next_state
    return reward_sum / n_starts


def quantile_huber_loss_f(quantiles, samples, device):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = T.abs(pairwise_delta)
    huber_loss = T.where(abs_pairwise_delta > 1,
                            abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = T.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (T.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss

def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False

def _random_seed(seed):
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)

# model save functions
def _save_model(net, dirpath):
    T.save(net.state_dict(), dirpath)

# model load functions
def _load_model(net, dirpath):
    net.load_state_dict(T.load(dirpath))