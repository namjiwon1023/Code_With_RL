import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

import ray

from networks import ActorSAC, CriticTwin

class Learner:
    def __init__(self, agent_args):
        self.agent_args = agent_args

        # actor-critic net setting
        self.actor = ActorSAC(self.agent_args).to(self.agent_args['learner_device'])
        self.critic = CriticTwin(self.agent_args).to(self.agent_args['learner_device'])

        self.critic_target = CriticTwin(self.agent_args).to(self.agent_args['learner_device'])
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Temperature Coefficient
        self.target_entropy = -self.agent_args['n_actions']
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.agent_args['learner_device'])

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.agent_args['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.agent_args['critic_lr'])
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.agent_args['alpha_lr'])

        # loss function
        self.criterion = nn.MSELoss()

        self.total_step = 0
        self.learning_step = 0

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def learn(self, buffer):
        if not buffer.ready.remote():
            return
        self.learning_step += 1

        # TD error
        # update value
        q1_loss, q2_loss, state = self._value_update(buffer)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss, new_log_prob = self._policy_update(state)

        # update Policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update Temperature Coefficient
        alpha_loss = self._temperature_update(new_log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # target network soft update
        if self.total_step % self.agent_args['target_update_interval'] == 0:
            self._target_soft_update(self.critic_target, self.critic, self.agent_args['tau'])

    def _value_update(self, buffer):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            state, next_state, action, reward, mask = self._get_batch_buffer(buffer)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic(state, action)
        # TD error
        # update value
        q1_loss = self.criterion(current_q1, target_q)
        q2_loss = self.criterion(current_q2, target_q)
        return q1_loss, q2_loss, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        q_1, q_2 = self.critic(state, new_action)
        q = T.min(q_1, q_2)
        # update actor network
        actor_loss = (self.alpha * new_log_prob - q).mean()
        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss

    def _get_batch_buffer(self, buffer):
        samples = ray.get(buffer.sample_batch.remote())
        # state = T.as_tensor(samples['state'], dtype=T.float32, device=self.agent_args['learner_device'])
        # next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.agent_args['learner_device'])
        # action = T.as_tensor(samples['action'], dtype=T.float32, device=self.agent_args['learner_device']).reshape(-1, self.agent_args['n_actions'])
        # reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.agent_args['learner_device']).reshape(-1, 1)
        # mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.agent_args['learner_device']).reshape(-1, 1)
        state = T.tensor(samples['state'], dtype=T.float32, device=self.agent_args['learner_device'])
        next_state = T.tensor(samples['next_state'], dtype=T.float32, device=self.agent_args['learner_device'])
        action = T.tensor(samples['action'], dtype=T.float32, device=self.agent_args['learner_device']).reshape(-1, self.agent_args['n_actions'])
        reward = T.tensor(samples['reward'], dtype=T.float32, device=self.agent_args['learner_device']).reshape(-1, 1)
        mask = T.tensor(samples['mask'], dtype=T.float32, device=self.agent_args['learner_device']).reshape(-1, 1)
        return state, next_state, action, reward, mask

    def weights_to_cpu(self, weights):
        return {k: v.cpu() for k, v in weights.items()}

    def weights_to_gpu(self, weights):
        return {k: v.cuda() for k, v in weights.items()}

    def get_weights(self):
        actor_weight = self.weights_to_cpu(self.actor.state_dict())
        critic_weight = self.weights_to_cpu(self.critic.state_dict())
        # print('actor_weight : ',actor_weight)
        return dict(actor_weights=actor_weight,
                    critic_weights=critic_weight)

    def set_weights(self, weights):
        actor_weight = self.weights_to_gpu(weights['actor_weights'])
        critic_weight = self.weights_to_gpu(weights['critic_weights'])
        self.actor.load_state_dict(actor_weight)
        self.critic.load_state_dict(critic_weight)

    def _target_soft_update(self, target_net, eval_net, tau):
        for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
            t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)


class Actor:
    def __init__(self, agent_args):
        self.agent_args = agent_args

        self.actor = ActorSAC(self.agent_args).to(self.agent_args['actor_device'])

        self.total_step = 0
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def select_test_action(self, state):
        with T.no_grad():
            # test_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.agent_args['actor_device']), evaluate=True, with_logprob=False)
            test_action, _ = self.actor(T.tensor(state, dtype=T.float32, device=self.agent_args['actor_device']), evaluate=True, with_logprob=False)
            test_action = test_action.detach().cpu().numpy()
        return test_action

    def select_exploration_action(self, state):
        with T.no_grad():
            if self.total_step <= self.agent_args['start_steps']:
                exploration_action = np.random.uniform(self.agent_args['low_action'], self.agent_args['max_action'], self.agent_args['n_actions'])
            else:
                # exploration_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.agent_args['actor_device']))
                exploration_action, _ = self.actor(T.tensor(state, dtype=T.float32, device=self.agent_args['actor_device']))
                exploration_action = exploration_action.detach().cpu().numpy()
        return exploration_action

    def get_weights(self):
        actor_weight = self.weights_to_cpu(self.actor.state_dict())
        return dict(actor_weights=actor_weight)

    def weights_to_cpu(self, weights):
        return {k: v.cpu() for k, v in weights.items()}

    def set_weights(self, weights):
        weights = self.weights_to_cpu(weights['actor_weights'])
        self.actor.load_state_dict(weights)

    def _evaluate_agent(self, env, agent, agent_args):
        reward_sum = 0
        for _ in range(agent_args['n_starts']):
            done = False
            state = env.reset()
            max_ep_len = env.spec.max_episode_steps
            ep_len = 0
            while not (done or (ep_len == max_ep_len)):
                ep_len += 1
                if agent_args['render']:
                    env.render()
                with eval_mode(agent):
                    action = agent.select_test_action(state)
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                state = next_state
        return reward_sum / agent_args['n_starts']

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
