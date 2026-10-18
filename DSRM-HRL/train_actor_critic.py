from tqdm import tqdm
from time import time
import torch
# 禁用Flash Attention来避免梯度计算问题
import os
os.environ['TORCH_USE_FLASH_ATTENTION'] = '0'
try:
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
except:
    pass

from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
# import matplotlib.pyplot as plt

from model.agent import *
from model.policy import *
from model.critic import *
from model.buffer import *
from env import KREnvironment_WholeSession_GPU

import utils
# from analyze_utils import analyze_lifecycle_ratios  # 导入分析函数


if __name__ == '__main__':
    
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--env_class', type=str, required=True, help='Environment class.')
    init_parser.add_argument('--policy_class', type=str, required=True, help='Policy class')
    init_parser.add_argument('--critic_class', type=str, required=True, help='Critic class')
    init_parser.add_argument('--agent_class', type=str, required=True, help='Learning agent class')
    init_parser.add_argument('--buffer_class', type=str, required=True, help='Buffer class.')
    
    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)
    envClass = eval('{0}.{0}'.format(initial_args.env_class))
    policyClass = eval('{0}.{0}'.format(initial_args.policy_class))
    criticClass = eval('{0}.{0}'.format(initial_args.critic_class))
    agentClass = eval('{0}.{0}'.format(initial_args.agent_class))
    bufferClass = eval('{0}.{0}'.format(initial_args.buffer_class))
    
    # experimental control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device number; set to -1 (default) if using cpu')
    
    # customized args
    parser = envClass.parse_model_args(parser)
    parser = policyClass.parse_model_args(parser)
    parser = criticClass.parse_model_args(parser)
    parser = agentClass.parse_model_args(parser)
    parser = bufferClass.parse_model_args(parser)
    args, _ = parser.parse_known_args()
    
    if args.cuda >= 0 and torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    utils.set_random_seed(args.seed)
    
    # Environment
    print("Loading environment")
    env = envClass(args)
    
    # Policy, Critic, Buffer, Agent
    print("Setup policy:")
    policy = policyClass(args, env)
    policy.to(device)
    print(policy)
    print("Setup critic:")
    if initial_args.agent_class == 'TD3':
        critic1 = criticClass(args, env, policy)
        critic1.to(device)
        critic2 = criticClass(args, env, policy)
        critic2.to(device)
        critic = [critic1, critic2]
    else:
        critic = criticClass(args, env, policy)
        critic.to(device)
    print(critic)
    print("Setup buffer:")
    buffer = bufferClass(args, env, policy, critic)
    print(buffer)
    print("Setup agent:")
    agent = agentClass(args, env, policy, critic, buffer)


    # 检查 agent 是否需要 critic 参数
    # import inspect
    # agent_signature = inspect.signature(agentClass.__init__)
    # agent_params = list(agent_signature.parameters.keys())
    
    # if 'critic' in agent_params:
    #     agent = agentClass(args, env, policy, critic, buffer)
    # else:
    #     agent = agentClass(args, env, policy, buffer)
    # print(agent)

    print(agent)
    
    # # 创建保存分析结果的目录
    # analysis_dir = os.path.join(args.save_path, '../analysis')
    # os.makedirs(analysis_dir, exist_ok=True)
    
    # # 定义分析函数
    # def analyze_and_save(agent, step):
    #     # 获取训练历史中的比例数据
    #     early_ratios = agent.training_history['early_stage_ratio']
    #     mature_ratios = agent.training_history['mature_stage_ratio']
    #     decline_ratios = agent.training_history['decline_stage_ratio']
        
    #     # 绘制比例变化图
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(early_ratios, label='Early Stage', color='green')
    #     plt.plot(mature_ratios, label='Mature Stage', color='blue')
    #     plt.plot(decline_ratios, label='Decline Stage', color='red')
        
    #     plt.xlabel('Training Steps')
    #     plt.ylabel('Ratio in Recommendation List')
    #     plt.title(f'Distribution of Video Lifecycle Stages (Step {step})')
    #     plt.legend()
    #     plt.grid(True)
        
    #     # 保存图片
    #     plt.savefig(os.path.join(analysis_dir, f'lifecycle_ratios_step_{step}.png'))
    #     plt.close()
        
    #     # 保存最近的统计数据
    #     window_size = min(1000, len(early_ratios))
    #     stats = {
    #         'step': step,
    #         'early_stage': np.mean(early_ratios[-window_size:]),
    #         'mature_stage': np.mean(mature_ratios[-window_size:]),
    #         'decline_stage': np.mean(decline_ratios[-window_size:])
    #     }
        
    #     # 将统计数据写入文件
    #     with open(os.path.join(analysis_dir, 'lifecycle_stats.txt'), 'a') as f:
    #         f.write(f"\nStep {step}:\n")
    #         f.write(f"Early stage: {stats['early_stage']:.3f}\n")
    #         f.write(f"Mature stage: {stats['mature_stage']:.3f}\n")
    #         f.write(f"Decline stage: {stats['decline_stage']:.3f}\n")
    #         f.write("-" * 50 + "\n")
        
    #     return stats
    
    # online training
    try:
        print(args)
        
        # # 修改agent的train方法来包含定期分析
        # original_step_train = agent.step_train
        
        # def new_step_train(*args, **kwargs):
        #     result = original_step_train(*args, **kwargs)
            
        #     # 每1000步进行一次分析
        #     if agent.update_step % 1000 == 0:
        #         print(f"\nAnalyzing lifecycle ratios at step {agent.update_step}...")
        #         stats = analyze_and_save(agent, agent.update_step)
        #         print(f"Current ratios - Early: {stats['early_stage']:.3f}, "
        #               f"Mature: {stats['mature_stage']:.3f}, "
        #               f"Decline: {stats['decline_stage']:.3f}")
            
        #     return result
        
        # # 替换agent的step_train方法
        # agent.step_train = new_step_train
        
        # # 开始训练
        agent.train()
        
        # # 训练结束后进行最终分析
        # print("\nPerforming final analysis...")
        # final_stats = analyze_and_save(agent, agent.update_step)
        # print("Training completed. Final lifecycle stage ratios:")
        # print(f"Early stage: {final_stats['early_stage']:.3f}")
        # print(f"Mature stage: {final_stats['mature_stage']:.3f}")
        # print(f"Decline stage: {final_stats['decline_stage']:.3f}")
        
    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time() + ' ' + '-' * 20)
            exit(1)
        # else:
        #     # 执行最终分析
        #     print("\nPerforming final analysis before exit...")
        #     final_stats = analyze_and_save(agent, agent.update_step)
        #     print("Final lifecycle stage ratios:")
        #     print(f"Early stage: {final_stats['early_stage']:.3f}")
        #     print(f"Mature stage: {final_stats['mature_stage']:.3f}")
        #     print(f"Decline stage: {final_stats['decline_stage']:.3f}")
            