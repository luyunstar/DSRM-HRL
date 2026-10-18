import torch
import numpy as np
import pandas as pd

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *
from torch.distributions import MultivariateNormal, Categorical
import torch.nn as nn
import torch.nn.functional as F

from diffusion.denoising_diffusion import UserHistoryDenoiser
from model.pre_lifetime.stl_2_itrans import predict_for_date as predict_stl_itrans

class OneStagePolicy_LHP1_d1(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument('--policy_action_hidden', type=int, nargs='+', default=[128])
        parser.add_argument('--action_std_init', type=float, default=0.1)
        parser.add_argument('--has_continuous_action_space', type=bool, default=True)
        
        parser.add_argument('--denoiser_model_path', type=str, default='/home/cigit_ly/KuaiSim-V2/code/diffusion/diffusion_model_50/history_denoiser_final.pth')
        parser.add_argument('--denoiser_embed_dim', type=int, default=128)
        parser.add_argument('--denoiser_n_head', type=int, default=4)
        parser.add_argument('--denoiser_n_layer', type=int, default=4)
        parser.add_argument('--denoiser_dropout', type=float, default=0.1)
        parser.add_argument('--denoiser_steps', type=int, default=50)

        return parser

    def __init__(self, args, environment):
        super().__init__(args, environment) 
        
        self.h_hyper_action_dim = 5
        self.l_hyper_action_dim = self.item_num
        self.h_action_dim = self.h_hyper_action_dim
        self.l_action_dim = self.l_hyper_action_dim
        self.effect_action_dim = self.slate_size
        self.has_continuous_action_space = args.has_continuous_action_space

        self.h_action_layer = nn.Sequential(
            DNN(self.state_dim, args.policy_action_hidden, self.h_action_dim,
                dropout_rate=self.dropout_rate, do_batch_norm=True),
            nn.Softplus()
        )
        self.l_action_layer = nn.Sequential(
            DNN(self.state_dim + self.h_hyper_action_dim, args.policy_action_hidden, self.l_action_dim,
                dropout_rate=self.dropout_rate, do_batch_norm=True),
            nn.Softmax(dim=-1)
        )
        self.h_action_std_init = args.action_std_init
        self.l_action_std_init = args.action_std_init

        path = f"/home/cigit_ly/KuaiSim-V2/code/dataset/KuaiRec/"
        self.user_pop_ratios = torch.tensor(pd.read_csv(path + 'user_pop_ratio.csv').to_numpy()).to(self.device)
        item_types = torch.tensor(pd.read_csv(path + 'item_types.csv').to_numpy()).to(self.device)
        item_types = item_types.flatten()
        if item_types.shape[0] != self.item_num:
            if item_types.shape[0] > self.item_num:
                item_types = item_types[:self.item_num]
            else:
                padding = torch.full((self.item_num - item_types.shape[0],), item_types[-1].item()).to(self.device)
                item_types = torch.cat([item_types, padding])
        self.item_types = item_types


        print("Initializing and loading pre-trained User History Denoiser...")
        self.user_history_denoiser = UserHistoryDenoiser(
            n_items=self.item_num,
            embed_dim=args.denoiser_embed_dim,
            n_head=args.denoiser_n_head,
            n_layer=args.denoiser_n_layer,
            dropout=args.denoiser_dropout,
            steps=args.denoiser_steps
        ).to(self.device)
        
        try:
            self.user_history_denoiser.denoise_fn.load_state_dict(torch.load(args.denoiser_model_path, map_location=self.device))
            print(f"Successfully loaded denoiser model from {args.denoiser_model_path}")
        except Exception as e:
            print(f"Error loading denoiser model: {e}. Using randomly initialized weights.")
            
        self.user_history_denoiser.eval()
        for param in self.user_history_denoiser.parameters():
            param.requires_grad = False
        print("User History Denoiser is frozen and in evaluation mode.")

    def _define_params(self, args, reader_stats):
        super()._define_params(args, reader_stats)
        self.state_dim = args.denoiser_embed_dim
        
    def get_user_state(self, observation):
        history_ids = observation['user_history']['history']
        denoised_user_state = self.user_history_denoiser.denoise(history_ids)
        state_dict = {'state': denoised_user_state, 'reg': 0.0}
        return state_dict, state_dict
    
    def calculate_lifecycle_metrics(self, item_ids, item_popularity):
        B, K = item_ids.shape
        metrics = {}
        item_lifecycle = self.item_lifecycle[item_ids]
        popular_mask = (item_popularity.gather(1, item_ids) >= 0.5)
        non_popular_mask = ~popular_mask
        rising_mask = (item_lifecycle == 0)
        mature_mask = (item_lifecycle == 1)
        decline_mask = (item_lifecycle == 2)
        non_pop_rising_mature = (non_popular_mask & (rising_mask | mature_mask)).sum(dim=1).float()
        non_pop_total = non_popular_mask.sum(dim=1).clamp(min=1e-5)
        metrics['non_pop_rising_mature_ratio'] = (non_pop_rising_mature / non_pop_total).mean()
        pop_decline = (popular_mask & decline_mask).sum(dim=1).float()
        pop_total = popular_mask.sum(dim=1).clamp(min=1e-5)
        metrics['pop_decline_ratio'] = (pop_decline / pop_total).mean()
        rising_count = rising_mask.sum(dim=1).float()
        mature_count = mature_mask.sum(dim=1).float()
        decline_count = decline_mask.sum(dim=1).float()
        metrics['life_health_score'] = ((rising_count + mature_count - decline_count) / K).mean()
        return metrics

    def set_init_var(self):
        h_action_var = torch.full((self.h_action_dim,), self.h_action_std_init * self.h_action_std_init,
                                  requires_grad=True).to(self.device)
        self.h_action_var = nn.Parameter(h_action_var)
        l_action_var = torch.full((self.l_action_dim,), self.l_action_std_init * self.l_action_std_init,
                                  requires_grad=True).to(self.device)
        self.l_action_var = nn.Parameter(l_action_var)

    def generate_action(self, h_state_dict, l_state_dict, feed_dict):
        h_state = h_state_dict['state']
        l_state = l_state_dict['state']
        user_pop_prefer = feed_dict['observation']['user_history']['user_pop_prefer']
        B = h_state.shape[0]
        h_action_mean = self.h_action_layer(h_state)
        h_cov_mat = nn.functional.relu(torch.diag(self.h_action_var).unsqueeze(dim=0)) + 1e-5
        dist = MultivariateNormal(h_action_mean, h_cov_mat)
        h_action = dist.sample()
        h_action_log_prob = dist.log_prob(h_action)
        with torch.no_grad():
            fair_weights = F.softmax(h_action[:, :2], dim=-1)
            life_weights = F.softmax(h_action[:, 2:], dim=-1)
            life_weights[:, 2] = life_weights[:, 2] * 0.3
            h_action = torch.cat([fair_weights, life_weights], dim=-1)
        l_state = torch.cat((l_state, h_action.detach()), dim=-1)
        l_action_mean = self.l_action_layer(l_state)
        item_popularity = self.item_types.unsqueeze(0).expand(B, -1).float()
        item_lifecycle = self.item_lifecycle.unsqueeze(0).expand(B, -1)
        rising_mask = (item_lifecycle == 0).float()
        mature_mask = (item_lifecycle == 1).float()
        decline_mask = (item_lifecycle == 2).float()
        base_scores = (l_action_mean * item_popularity * h_action[:, 0].view(-1, 1) + l_action_mean * (1 - item_popularity) * h_action[:, 1].view(-1, 1))
        life_boost = (rising_mask * h_action[:, 2].view(-1, 1) + mature_mask * h_action[:, 3].view(-1, 1) + decline_mask * h_action[:, 4].view(-1, 1))
        final_scores = base_scores * (1.0 + life_boost)
        non_popular_boost = ((1 - item_popularity) * (rising_mask + mature_mask))
        final_scores += final_scores * non_popular_boost * 0.5
        final_scores = final_scores / (final_scores.sum(dim=1, keepdim=True) + 1e-10)
        final_scores = torch.clamp(final_scores, min=0.0, max=1.0)
        dist = Categorical(final_scores)
        l_action = torch.multinomial(final_scores, num_samples=self.effect_action_dim, replacement=False)
        l_action_log_probs = []
        for i in range(self.effect_action_dim):
            l_action_log_probs.append(dist.log_prob(l_action[:, i]))
        l_action_log_prob = torch.stack(l_action_log_probs, dim=1).mean(dim=1)
        unpopular_ratios = []
        for i in range(B):
            batch_unpopular_mask = (1 - item_popularity[i]).bool()
            batch_unpopular_items = torch.where(batch_unpopular_mask)[0]
            batch_selected_unpopular = torch.isin(l_action[i], batch_unpopular_items)
            unpopular_ratios.append(batch_selected_unpopular.float().mean())
        unpopular_item_ratio = torch.stack(unpopular_ratios)
        fair_weight = h_action[:, 1]
        early_ratio = rising_mask.mean(dim=1)
        mature_ratio = mature_mask.mean(dim=1)
        decline_ratio = decline_mask.mean(dim=1)
        lifecycle_ratios = torch.stack([early_ratio, mature_ratio, decline_ratio], dim=1)
        out_dict = {'h_action': h_action, 'l_action': l_action, 'h_action_log_prob': h_action_log_prob, 'l_action_log_prob': l_action_log_prob, 'indices': l_action, 'user_pop_prefer': user_pop_prefer, 'unpopular_item_ratio': unpopular_item_ratio, 'fair_weight': fair_weight, 'lifecycle_ratios': lifecycle_ratios, 'early_ratio': early_ratio, 'mature_ratio': mature_ratio, 'decline_ratio': decline_ratio, 'reg': self.get_regularization(self.h_action_layer) + self.get_regularization(self.l_action_layer)}
        return out_dict

    def get_score(self, hyper_action, candidate_item_enc, item_dim):
        return linear_scorer(hyper_action, candidate_item_enc, item_dim)

    def evaluate(self, feed_dict, user_pop_prefer):
        h_state = feed_dict['h_state'].view(-1, self.state_dim)
        l_state = feed_dict['l_state'].view(-1, self.state_dim)
        B = h_state.shape[0]
        h_action = feed_dict['h_action'].view(-1, self.h_action_dim)
        l_action = feed_dict['l_action'].view(-1, self.effect_action_dim)
        h_action_mean = self.h_action_layer(h_state)
        h_cov_mat = nn.functional.relu(torch.diag(self.h_action_var).unsqueeze(dim=0)) + 1e-5
        dist = MultivariateNormal(h_action_mean, h_cov_mat)
        h_action_log_prob = dist.log_prob(h_action)
        h_dist_entropy = dist.entropy()
        with torch.no_grad():
            fair_weights = F.softmax(h_action[:, :2], dim=-1)
            life_weights = F.softmax(h_action[:, 2:], dim=-1)
            life_weights[:, 2] = life_weights[:, 2] * 0.3
            h_action_constrained = torch.cat([fair_weights, life_weights], dim=-1)
        l_state = torch.cat((l_state, h_action_constrained.detach()), dim=-1)
        l_action_mean = self.l_action_layer(l_state)
        item_popularity = self.item_types.unsqueeze(0).expand(B, -1).float()
        item_lifecycle = self.item_lifecycle.unsqueeze(0).expand(B, -1)
        rising_mask = (item_lifecycle == 0).float()
        mature_mask = (item_lifecycle == 1).float()
        decline_mask = (item_lifecycle == 2).float()
        base_scores = (l_action_mean * item_popularity * h_action_constrained[:, 0].view(-1, 1) + l_action_mean * (1 - item_popularity) * h_action_constrained[:, 1].view(-1, 1))
        life_boost = (rising_mask * h_action_constrained[:, 2].view(-1, 1) + mature_mask * h_action_constrained[:, 3].view(-1, 1) + decline_mask * h_action_constrained[:, 4].view(-1, 1))
        final_scores = base_scores * (1.0 + life_boost)
        non_popular_boost = ((1 - item_popularity) * (rising_mask + mature_mask))
        final_scores += final_scores * non_popular_boost * 0.5
        final_scores = final_scores / (final_scores.sum(dim=1, keepdim=True) + 1e-10)
        final_scores = torch.clamp(final_scores, min=0.0, max=1.0)
        dist = Categorical(final_scores)
        l_action_log_probs = []
        for i in range(self.effect_action_dim):
            l_action_log_probs.append(dist.log_prob(l_action[:, i]))
        l_action_log_prob = torch.stack(l_action_log_probs, dim=1).mean(dim=1)
        l_dist_entropy = dist.entropy()
        return h_action_log_prob, h_dist_entropy, l_action_log_prob, l_dist_entropy

    def get_forward(self, feed_dict: dict):
        observation = feed_dict['observation']
        h_state_dict, l_state_dict = self.get_user_state(observation)
        out_dict = self.generate_action(h_state_dict, l_state_dict, feed_dict)
        out_dict['h_state'] = h_state_dict['state']
        out_dict['l_state'] = l_state_dict['state']
        out_dict['reg'] = out_dict['reg']
        return out_dict

    def forward(self, feed_dict: dict, return_prob=True):
        return self.get_forward(feed_dict)

