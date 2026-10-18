import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseOnPolicyRLAgent import BaseOnPolicyRLAgent
import pandas as pd
from torch import nn
from model.pre_lifetime.trans import predict_for_date as predict_stl_itrans

class LHP1(BaseOnPolicyRLAgent):
    @staticmethod
    def parse_model_args(parser):
        parser = BaseOnPolicyRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-4)
        parser.add_argument('--critic_decay', type=float, default=1e-4)
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01)
        parser.add_argument('--train_epoch_num', type=int, default=4)
        parser.add_argument('--eps_clip', type=float, default=0.8)
        return parser

    def __init__(self, *input_args):
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        path = f"/home/cigit_ly/KuaiSim-V2/code/dataset/KuaiRec/"
        self.user_pop_ratios = torch.tensor(pd.read_csv(path + 'user_pop_ratio.csv').to_numpy()).to(self.device)
        item_types_data = torch.tensor(pd.read_csv(path + 'item_types.csv').to_numpy()).to(self.device)
        self.item_types = item_types_data.flatten()
        
        if hasattr(self.actor, 'item_num'):
            target_size = self.actor.item_num
        elif hasattr(self.actor, 'l_hyper_action_dim'):
            target_size = self.actor.l_hyper_action_dim
        else:
            target_size = self.item_types.shape[0]
            
        if self.item_types.shape[0] != target_size:
            if self.item_types.shape[0] > target_size:
                self.item_types = self.item_types[:target_size]
            else:
                padding = torch.full((target_size - self.item_types.shape[0],), self.item_types[-1].item()).to(self.device)
                self.item_types = torch.cat([self.item_types, padding])
        

        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        self.train_epoch_num = args.train_epoch_num
        self.eps_clip = args.eps_clip
        self.update_step = 0
        self.MseLoss = nn.MSELoss()

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)

        print("Optimizer setup: Training actor's action layers and critic ONLY.")
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.h_action_layer.parameters(), 'lr': args.actor_lr},
            {'params': self.actor.l_action_layer.parameters(), 'lr': args.actor_lr},
            {'params': self.actor.h_action_var, 'lr': args.actor_lr / 20},
            {'params': self.actor.l_action_var, 'lr': args.actor_lr / 20},
            {'params': self.critic.parameters(), 'lr': args.critic_lr},
        ])

        self.do_actor_update = True
        self.do_critic_update = True

        self.registered_models.append((self.actor, self.optimizer, "_actor"))
        self.registered_models.append((self.critic, self.optimizer, "_critic"))

    def setup_monitors(self):
        super().setup_monitors()
        self.training_history.update({
            'h_actor_loss': [], 'l_actor_loss': [], 'critic_loss': [], 'dist_entropy_loss': [],
            'H_V': [], 'next_H_V': [], 'L_V': [], 'next_L_V': [], 'life_reward': [],
            'non_pop_rising_mature_ratio': [], 'pop_decline_ratio': [], 'life_health_score': []
        })

    def step_train(self):
        observation, policy_output, user_feedback, done_mask, n_observation = self.buffer.sample()
        h_old_log_prob = policy_output['h_action_log_prob'].view(-1)
        l_old_log_prob = policy_output['l_action_log_prob'].view(-1)
        reward = user_feedback['reward'].view(-1)
        total_len = policy_output['h_state'].shape[0]
        unpopular_item_ratio = policy_output['unpopular_item_ratio'].view(-1)
        is_train = True
        
        item_ids = policy_output['l_action'].long()
        lifecycle_stages = self.item_lifecycle[item_ids]
        B = policy_output['l_action'].shape[0]
        item_popularity_batch = self.item_types.unsqueeze(0).expand(B, -1)
        lifecycle_metrics = self.actor.calculate_lifecycle_metrics(
            policy_output['l_action'].long(),
            item_popularity_batch
        )

        for _ in range(self.train_epoch_num):
            idxes = np.arange(total_len)
            np.random.shuffle(idxes)
            
            for i in range(0, total_len, self.batch_size):
                batch_idx = idxes[i:i+self.batch_size]
                idx = torch.tensor(batch_idx).to(self.device)

                current_observation = {'user_profile': {k: v[idx] for k, v in observation['user_profile'].items()},
                                       'user_history': {k: v[idx] for k, v in observation['user_history'].items()}}
                current_policy_output = {k: v[idx] for k, v in policy_output.items()}
                current_h_old_log_prob = h_old_log_prob[idx]
                current_l_old_log_prob = l_old_log_prob[idx]
                current_done = done_mask[idx]
                current_reward = reward[idx]
                current_unpopular_item_ratio = unpopular_item_ratio[idx]
                
                user_pop_prefer = current_observation['user_history']['user_pop_prefer'].reshape(-1, 1)
                
                current_lifecycle = lifecycle_stages[idx]
                item_id = current_policy_output['l_action'].long()
                item_popularity = self.item_types[item_id]
                
                rising_mask = (current_lifecycle == 0).float()
                mature_mask = (current_lifecycle == 1).float()
                decline_mask = (current_lifecycle == 2).float()
                
                life_reward = (rising_mask * (1 - item_popularity) * 0.1 +
                               mature_mask * (1 - item_popularity) * 0.05 -
                               decline_mask * item_popularity * 0.2).mean(dim=1)
                
                h_current_reward = current_reward  + current_unpopular_item_ratio / (1 + user_pop_prefer.squeeze()) * 0.5
                l_current_reward = current_reward

                next_observation = {'user_profile': {k: v[idx] for k, v in n_observation['user_profile'].items()},
                                    'user_history': {k: v[idx] for k, v in n_observation['user_history'].items()}}

                h_log_prob, h_dist_entropy, l_log_prob, l_dist_entropy = self.actor.evaluate(current_policy_output, user_pop_prefer)
                
                h_ratios = torch.exp(h_log_prob - current_h_old_log_prob.detach())
                l_ratios = torch.exp(l_log_prob - current_l_old_log_prob.detach())

                current_critic_output = self.apply_critic(current_observation, current_policy_output, self.critic)
                current_h_state_values = current_critic_output['h_v']
                current_l_state_values = current_critic_output['l_v']

                with torch.no_grad():
                    next_policy_output = self.apply_policy(next_observation, self.actor_target, 0., False, is_train)
                    target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
                    next_h_state_values = target_critic_output['h_v']
                    next_l_state_values = target_critic_output['l_v']
                
                target_h_state_values = self.gamma * torch.squeeze(next_h_state_values) * (~current_done) + h_current_reward
                target_l_state_values = self.gamma * torch.squeeze(next_l_state_values) * (~current_done) + l_current_reward

                h_advantages = target_h_state_values - current_h_state_values
                l_advantages = target_l_state_values - current_l_state_values

                h_surr1 = h_ratios * h_advantages.detach()
                h_surr2 = torch.clamp(h_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * h_advantages.detach()
                l_surr1 = l_ratios * l_advantages.detach()
                l_surr2 = torch.clamp(l_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * l_advantages.detach()

                h_actor_loss = -torch.min(h_surr1, h_surr2)
                l_actor_loss = -torch.min(l_surr1, l_surr2)
                critic_loss = self.MseLoss(current_h_state_values, target_h_state_values.detach()) + \
                              self.MseLoss(current_l_state_values, target_l_state_values.detach())
                dist_entropy_loss = (-h_dist_entropy - l_dist_entropy) * 0.0005
                
                loss = (h_actor_loss + l_actor_loss).mean() + critic_loss + dist_entropy_loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10, norm_type=2)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        loss_dict = {
            'h_actor_loss': h_actor_loss.mean().item(), 'l_actor_loss': l_actor_loss.mean().item(),
            'critic_loss': critic_loss.item(), 'dist_entropy_loss': dist_entropy_loss.mean().item(),
            'H_V': torch.mean(current_h_state_values).item(), 'L_V': torch.mean(current_l_state_values).item(),
            'next_H_V': torch.mean(next_h_state_values).item(), 'next_L_V': torch.mean(next_l_state_values).item(),
            'life_reward': life_reward.mean().item()
        }
        loss_dict.update({k: v.item() for k, v in lifecycle_metrics.items()})

        for k, v in loss_dict.items():
            if k in self.training_history:
                self.training_history[k].append(v)
        return loss_dict

    def apply_policy(self, observation, actor, *policy_args):
        epsilon, do_explore, is_train = policy_args
        input_dict = {'observation': observation, 'candidates': self.env.get_candidate_info(observation),
                      'epsilon': epsilon, 'do_explore': do_explore, 'is_train': is_train, 'batch_wise': False}
        return actor(input_dict)

    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'h_state': policy_output['h_state'], 'l_state': policy_output['l_state'],
                     'user_pop_prefer': observation['user_history']['user_pop_prefer'].reshape(-1, 1),
                     'h_action': policy_output['h_action']}
        return critic(feed_dict)

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

