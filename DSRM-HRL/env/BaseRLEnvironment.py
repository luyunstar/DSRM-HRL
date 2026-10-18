from argparse import Namespace
import torch
import json

import utils
from model.simulator import *
from reader import *


class BaseRLEnvironment():
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - max_step_per_episode
        - initial_temper
        '''
        parser.add_argument('--max_step_per_episode', type=int, default=100,
                            help='max number of iteration allowed in each episode')
        parser.add_argument('--initial_temper', type=float, default=10, help='initial temper of users')
        parser.add_argument('--user_pop_prefer_path', type=str, default='output/user_pop_prefer.txt',
                        help='Path to user_pop_prefer file')
        # # 添加预期的 prefer_dim 参数
        # parser.add_argument('--prefer_dim', type=int, default=902,
        #                     help='Expected dimension of user_pop_prefer vectors')

        
        return parser

    def __init__(self, args):
        self.device = args.device
        super().__init__()
        self.max_step_per_episode = args.max_step_per_episode
        self.initial_temper = args.initial_temper
        # # 添加默认值定义，根据你的实际维度需求调整size
        # self.default_pop_prefer = None  
        # # 如果需要batch维度，会通过broadcast自动扩展

        # # self.user_pop_prefer = self.load_user_pop_prefer(args.user_pop_prefer_path)
        # self.user_pop_prefer = self.load_user_pop_prefer(args.user_pop_prefer_path, args.prefer_dim)

    # def load_user_pop_prefer(self, file_path, expected_prefer_dim):
    #     """
    #     从指定的文件路径加载 user_pop_prefer 数据。

    #     Args:
    #         file_path (str): user_pop_prefer 文件的路径。
    #         expected_prefer_dim (int): 模型预期的偏好向量维度。

    #     Returns:
    #         torch.Tensor: 一个张量，形状为 (num_users, prefer_dim)。
    #     """
    #     try:
    #         with open(file_path, 'r') as f:
    #             data = json.load(f)
    #     except FileNotFoundError:
    #         print(f"文件 {file_path} 未找到。使用默认偏好值。")
    #         # 如果文件未找到，使用预期维度初始化默认偏好
    #         self.default_pop_prefer = torch.zeros(1, expected_prefer_dim).to(self.device)
    #         return self.default_pop_prefer

    #     user_ids = [int(k) for k in data.keys()]
    #     max_user_id = max(user_ids)
    #     sample_prefer = data[str(user_ids[0])]
    #     actual_prefer_dim = len(sample_prefer)

    #     if actual_prefer_dim != expected_prefer_dim:
    #         print(f"警告: 数据中的 prefer_dim ({actual_prefer_dim}) 与预期的 ({expected_prefer_dim}) 不一致。进行截断或填充。")
    #         if actual_prefer_dim > expected_prefer_dim:
    #             # 截断偏好向量
    #             data = {k: v[:expected_prefer_dim] for k, v in data.items()}
    #             prefer_dim = expected_prefer_dim
    #         else:
    #             # 填充偏好向量
    #             padding = [0.0] * (expected_prefer_dim - actual_prefer_dim)
    #             data = {k: v + padding for k, v in data.items()}
    #             prefer_dim = expected_prefer_dim
    #     else:
    #         prefer_dim = actual_prefer_dim

    #     # 初始化 default_pop_prefer
    #     self.default_pop_prefer = torch.zeros(1, prefer_dim).to(self.device)

    #     # 初始化一个张量来存储所有用户的偏好
    #     user_pop_prefer_tensor = torch.zeros(max_user_id + 1, prefer_dim).to(self.device)

    #     for k, v in data.items():
    #         uid = int(k)
    #         user_pop_prefer_tensor[uid] = torch.tensor(v, dtype=torch.float).to(self.device)

    #     return user_pop_prefer_tensor
    
    def reset(self, params):
        pass

    def step(self, action):
        pass

    def get_user_model(self, log_path, device, from_load=True):
        infile = open(log_path, 'r')
        class_args = eval(infile.readline())  # example: Namespace(model='KRMBUserResponse', reader='KRMBSeqReader')
        model_args = eval(infile.readline())  # model parameters in Namespace
        infile.close()
        print(model_args.model_path)
        checkpoint = torch.load(model_args.model_path + ".checkpoint", map_location=device)
        reader_stats = checkpoint["reader_stats"]
        modelClass = eval('{0}.{0}'.format(class_args.model))
        model = modelClass(model_args, reader_stats, device)
        if from_load:
            model.load_from_checkpoint(model_args.model_path, with_optimizer=False)
        model = model.to(device)
        return reader_stats, model, model_args

    def get_reader(self, log_path):
        infile = open(log_path, 'r')
        class_args = eval(infile.readline())  # example: Namespace(model='KRMBUserResponse', reader='KRMBSeqReader')
        training_args = eval(infile.readline())  # model parameters in Namespace
        training_args.val_holdout_per_user = 0
        training_args.test_holdout_per_user = 0
        training_args.device = self.device
        infile.close()
        readerClass = eval('{0}.{0}'.format(class_args.reader))
        reader = readerClass(training_args)
        return reader, training_args

    def get_observation_from_batch(self, sample_batch):
        '''
        extract observation from the reader's sample batch
        @input:
        - sample_batch: {
            'user_id': (B,)
            'uf_{feature}': (B,F_dim(feature)), user features
            'history': (B,max_H)
            'history_length': (B,)
            'history_if_{feature}': (B, max_H * F_dim(feature))
            'history_{response}': (B, max_H)
            ... user ground truth feedbacks are not included as observation
        }
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        '''
        sample_batch = utils.wrap_batch(sample_batch, device=self.device)
        profile = {'user_id': sample_batch['user_id']}
        for k, v in sample_batch.items():
            if 'uf_' in k:
                profile[k] = v
        
        # user_ids = sample_batch['user_id'].long()

        # if self.user_pop_prefer is not None:
        #     try:
        #         # 假设 user_ids 是从0开始的连续整数，可以直接索引
        #         user_pop_prefer = self.user_pop_prefer[user_ids]
        #     except IndexError:
        #         print("某些 user_id 超出了 user_pop_prefer 的范围，使用默认值。")
        #         user_pop_prefer = self.default_pop_prefer.expand(user_ids.size(0), -1)
        # else:
        #     user_pop_prefer = self.default_pop_prefer.expand(user_ids.size(0), -1)

        # history = {
        #     'history': sample_batch['history'], 
        #     'user_pop_prefer': user_pop_prefer.squeeze()
        # }
        
        
        history = {'history': sample_batch['history'], 
                   'user_pop_prefer': sample_batch['user_pop_prefer'].squeeze()}

        # history = {'history': sample_batch['history'], 
        #             'user_pop_prefer': sample_batch.get('user_pop_prefer', self.default_pop_prefer).squeeze()}
        for k, v in sample_batch.items():
            if 'history_' in k:
                history[k] = v
        return {'user_profile': profile, 'user_history': history}
