from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

# EXPANDED TRANSITION TUPLE: Now stores distinct env and comm fields
# If not using gumbel comm, comm_ fields will simply be stored as None
Transition = namedtuple('Transition', (
    'state', 
    'action', 'action_out', 'value', # Environment policy tracking
    'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc',
    'comm_action_out', 'comm_value'  # Communication policy tracking
))


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        
        # Flag to trigger the new Decoupled Gumbel Comm network
        self.use_gumbel_comm = getattr(args, 'use_gumbel_comm', False)
        
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]


    def get_episode(self, epoch):
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):
            misc = dict()
            # Old Hard Attention logic (only runs if old code path active)
            if t == 0 and self.args.hard_attn and self.args.commnet and not self.use_gumbel_comm:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                
                # --- FORWARD PASS CONDITIONAL ---
                if self.use_gumbel_comm:
                    payload, prev_hid = self.policy_net(x, info)
                    (action_out, value), (comm_action_out, comm_value) = payload
                else:
                    action_out, value, prev_hid = self.policy_net(x, info)
                    comm_action_out, comm_value = None, None

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if self.use_gumbel_comm:
                    payload = self.policy_net(x, info)
                    (action_out, value), (comm_action_out, comm_value) = payload
                else:
                    action_out, value = self.policy_net(x, info)
                    comm_action_out, comm_value = None, None

            # Environment Action Selection
            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            # Old hard attention logic preservation
            if self.args.hard_attn and self.args.commnet and not self.use_gumbel_comm:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            # Record transition (including new comm_ fields)
            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc, comm_action_out, comm_value)
            episode.append(trans)
            state = next_state
            if done:
                break
                
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions
        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # Retrieve Environment Values
        values = torch.cat(batch.value, dim=0).view(batch_size, n)
        
        # Retrieve Comm Values (if using Gumbel)
        if self.use_gumbel_comm:
            comm_values = torch.cat(batch.comm_value, dim=0).view(batch_size, n)
            comm_action_out_list = list(zip(*batch.comm_action_out)) 
            # List of tokens, each is a list of logits across the batch
            comm_action_logits = [torch.cat(t, dim=0) for t in comm_action_out_list]

        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        
        # Advantages for Env and Comm
        advantages = torch.Tensor(batch_size, n)
        if self.use_gumbel_comm:
            comm_advantages = torch.Tensor(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0

        # Calculate Returns (Shared Objective)
        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]
            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])

        # Calculate Advantages
        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]
            if self.use_gumbel_comm:
                # Comm Policy is evaluated against its own Critic
                comm_advantages[i] = returns[i] - comm_values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            if self.use_gumbel_comm:
                comm_advantages = (comm_advantages - comm_advantages.mean()) / (comm_advantages.std() + 1e-8)

        # --- ENVIRONMENT ACTION LOSS (Original Code) ---
        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            env_action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            env_action_loss *= alive_masks.unsqueeze(-1)
        else:
            env_action_loss = -advantages.view(-1) * log_prob.squeeze()
            env_action_loss *= alive_masks

        env_action_loss = env_action_loss.sum()
        stat['action_loss'] = env_action_loss.item()

        # Env Value Loss
        targets = returns
        env_value_loss = (values - targets).pow(2).view(-1)
        env_value_loss *= alive_masks
        env_value_loss = env_value_loss.sum()
        stat['value_loss'] = env_value_loss.item()

        # Base Loss
        loss = env_action_loss + self.args.value_coeff * env_value_loss

        # --- NEW: COMMUNICATION GRADIENTS & PENALTIES ---
        if self.use_gumbel_comm:
            comm_action_loss = 0
            comm_entropy = 0
            token_penalty_loss = 0
            
            # For each token in the phrase
            for token_logits in comm_action_logits:
                # Calculate Log Probs for the token distributions
                token_log_probs = F.log_softmax(token_logits, dim=-1)
                token_probs = F.softmax(token_logits, dim=-1)
                
                # We apply REINFORCE using the continuous probability distribution over words
                # Maximize prob of words chosen if advantage is positive
                # Multiply by Comm Policy's advantage
                adv = comm_advantages.view(-1).unsqueeze(-1)
                
                # Policy gradient step for comm action
                # Note: PyTorch backward auto-handles Gumbel gradients, but standard PPO/PG 
                # often explicitly calculates the log prob of the chosen action for stability.
                # Here we apply the advantage to the expected log prob.
                step_loss = -adv * token_log_probs
                step_loss = step_loss.sum(dim=-1) * alive_masks
                comm_action_loss += step_loss.sum()
                
                # Regularize with Entropy to encourage exploration of vocab
                comm_entropy -= (token_log_probs * token_probs).sum()
                
                # Token Sparsity Penalty: Penalize probabilities assigned to non-zero (non-silence) tokens
                # Assuming Token 0 is [SILENCE]
                if getattr(self.args, 'token_penalty', 0) > 0:
                    informational_token_probs = token_probs[:, 1:] # Everything except index 0
                    token_penalty_loss += informational_token_probs.sum() * self.args.token_penalty

            # Comm Value Loss
            comm_value_loss = (comm_values - targets).pow(2).view(-1)
            comm_value_loss *= alive_masks
            comm_value_loss = comm_value_loss.sum()
            
            stat['comm_action_loss'] = comm_action_loss.item()
            stat['comm_value_loss'] = comm_value_loss.item()
            stat['token_penalty'] = token_penalty_loss.item() if isinstance(token_penalty_loss, torch.Tensor) else token_penalty_loss
            
            # Combine all losses
            loss += comm_action_loss + self.args.value_coeff * comm_value_loss + token_penalty_loss
            
            if self.args.entr > 0:
                loss -= self.args.entr * comm_entropy

        # Old Env Entropy Regularization
        if not self.args.continuous and not self.use_gumbel_comm:
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        loss.backward()
        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)