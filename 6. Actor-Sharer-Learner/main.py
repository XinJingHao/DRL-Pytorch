import os
import time
import envpool
import argparse
import torch
import numpy as np
from datetime import datetime
import torch.multiprocessing as mp
from evaluator import eval_process
from recorder import record_process
from actor import actor_process
from learner import learner_process
from utils import str2bool
from AtariNames import Name
from multiprocessing.managers import BaseManager


if __name__ == '__main__':
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--EnvIdex', type=int, default=1, help='Index of Environment; Check AtariNames.py')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_train_steps', type=int, default=int(5e7), help='Max Total training steps')
    parser.add_argument('--eval_freq', type=int, default=int(5e3), help='Model evaluating freq, in bpsteps.')

    parser.add_argument('--eval_envs', type=int, default=1, help='number of envs for evaluation')
    parser.add_argument('--train_envs', type=int, default=128, help='number of envs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--TPS', type=int, default=8, help='Transitions Per Step; DQN nature: train 32 samples every 4 envsteps, thus 32/4=8')
    parser.add_argument('--time_feedback', type=str2bool, default=True, help='Whether use time feedback in actor')

    parser.add_argument('--A_dvc', type=str, default='cuda:0', help='Actor device')
    parser.add_argument('--B_dvc', type=str, default='cpu', help='Replay buffer device, either on cpu or L_dvc')
    parser.add_argument('--L_dvc', type=str, default='cuda:0', help='Learner device')
    parser.add_argument('--E_dvc', type=str, default='cuda:0', help='Evaluator device')

    parser.add_argument('--explore_steps', type=int, default=int(150e3), help='Random envsteps before tranning.')
    parser.add_argument('--buffersize', type=int, default=int(1e6), help='Replay Buffer size.')
    parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--fc_width', type=int, default=512, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=6.25e-5, help='Learning rate')
    parser.add_argument('--hard_update_freq', type=int, default=int(2E3), help='Hard target update frequency, in bpsteps')
    parser.add_argument('--init_explore_frac', type=float, default=1.0, help='init explore envs = 1.0*128')
    parser.add_argument('--decay_step', type=int, default=int(500e3), help='linear decay (env)steps for e-greedy noise')
    parser.add_argument('--end_explore_frac', type=float, default=0.032, help='end explore fraction = 0.032*128=4')
    parser.add_argument('--min_eps', type=float, default=0.01, help='minimal e-greedy noise')
    parser.add_argument('--upload_freq', type=int, default=int(50), help='learner update freq, in bpsteps')

    opt = parser.parse_args()
    opt.ExpEnvName = Name[opt.EnvIdex]+"-v5"
    spec = envpool.make_spec(opt.ExpEnvName)
    opt.action_dim = spec.action_space.n
    assert opt.buffersize >= opt.explore_steps

    if opt.B_dvc == 'cpu':
        from sharer import shared_data_cpu as shared_data
    else:
        from sharer import shared_data_cuda as shared_data
        assert opt.B_dvc == opt.L_dvc #使用GPUbuffer时，需要learner和buffer在同一设备上，以保证最大效率
    print(opt)

    # Set seed
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # Set write path
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    opt.writepath = 'runs/ASL' + '_S{}_big_{}'.format(opt.seed, opt.ExpEnvName) + timenow

    if not os.path.exists('runs'): os.mkdir('runs')
    mp.set_start_method('spawn')

    # Register sharer
    BaseManager.register("shared_data", callable=shared_data)  # 注册 Sharer，凡是注册到管理器中的类/对象，都可以被不同进程共享,"shared_data"是注册的名字，后面实例化时要调用这个名字
    ShareManager = BaseManager()
    ShareManager.start()
    shared_data = ShareManager.shared_data(opt)  # 实例化一个shared_data

    processes = []
    # actor process
    processes.append(mp.Process(target=actor_process, args=(opt, shared_data)))
    processes[-1].start()

    # learner process
    processes.append(mp.Process(target=learner_process, args=(opt, shared_data)))
    processes[-1].start()

    # evaluator process
    for _ in range(3):
        #建议多开几个，防止eval速度跟不上actor速度
        processes.append(mp.Process(target=eval_process, args=(_, opt, shared_data)))
        processes[-1].start()

    # recorder process
    processes.append(mp.Process(target=record_process, args=(opt, shared_data)))
    processes[-1].start()

    # 阻塞除recorder以外的所有进程
    for _ in range(len(processes)-1):
        processes[_].join()

    # 等待recorder工作结束后，结束进程
    time.sleep(120)
    processes[-1].terminate()













