import numpy as np
import torch
import gym
from DQN import DQN_Agent,ReplayBuffer_torch
import os, shutil
from datetime import datetime
import argparse
from utils import evaluate_policy, str2bool, LinearSchedule
from tianshou_wrappers import make_env_tianshou

EnvName = ['EnduroNoFrameskip-v4', 'PongNoFrameskip-v4']

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=1, help='Index of the Env')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=900, help='which model to load')

parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=1E6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1E5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='random steps befor trainning,5E4 in DQN Nature')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=1.0, help='explore noise')
parser.add_argument('--DDQN', type=str2bool, default=False, help='True:DDQN; False:DQN')
parser.add_argument('--buffersize', type=int, default=1e4, help='Size of the replay buffer, max 8e5')
parser.add_argument('--target_freq', type=int, default=1E3, help='frequency of target net updating')
parser.add_argument('--hardtarget', type=str2bool, default=True, help='True: update target net hardly(copy)')

parser.add_argument('--noop_reset', type=str2bool, default=False, help='use NoopResetEnv or not')
parser.add_argument('--huber_loss', type=str2bool, default=True, help='True: use huber_loss; False:use mse_loss')
parser.add_argument('--anneal_frac', type=int, default=3e5, help='annealing fraction of e-greedy nosise')
parser.add_argument('--FC_hidden', type=int, default=200, help='number of units in Fully Connected layer')
parser.add_argument('--train_freq', type=int, default=1, help='model trainning frequency')


opt = parser.parse_args()
print(opt)


def main():
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    env = make_env_tianshou(EnvName[opt.EnvIdex],seed = opt.seed, noop_reset = opt.noop_reset, episode_life = True, clip_rewards = True)
    eval_env = make_env_tianshou(EnvName[opt.EnvIdex],seed = opt.seed+1, noop_reset = opt.noop_reset, episode_life = True, clip_rewards = True)
    opt.action_dim = env.action_space.n

    #Use DDQN or DQN
    if opt.DDQN: algo_name = 'DDQN'
    else: algo_name = 'DQN'

    print('Algorithm:',algo_name,'  Env:',EnvName[opt.EnvIdex],'  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '\n')

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-7]
        timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2::]
        writepath = 'runs/S{}_{}_{}'.format(opt.seed,algo_name,EnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    model = DQN_Agent(opt)
    if opt.Loadmodel: model.load(algo_name,EnvName[opt.EnvIdex],opt.ModelIdex)

    #explore noise linearly annealed from 1.0 to 0.02 within 200k steps
    schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=0.02, initial_p=opt.exp_noise)
    model.exp_noise = opt.exp_noise

    if opt.render:
        score = evaluate_policy(eval_env, model,opt.render,20)
        print('EnvName:', EnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
    else:
        #build replay buffer
        buffer = ReplayBuffer_torch(max_size=int(opt.buffersize))

        #begin to iterate
        total_steps = -1
        while total_steps < opt.Max_train_steps:

            s, done = env.reset(), False
            while not done:
                a = model.select_action(s, evaluate=False)
                s_prime, r, done, info = env.step(a)
                buffer.add(s, a, r, s_prime, done)
                s = s_prime


                # train, e-decay, log, save
                if buffer.size >= opt.random_steps:
                    total_steps += 1
                    if total_steps % opt.train_freq == 0: model.train(buffer)

                    '''e-greedy decay'''
                    model.exp_noise = schedualer.value(total_steps)

                    '''record & log'''
                    if total_steps % opt.eval_interval == 0:
                        score = evaluate_policy(eval_env, model)
                        if opt.write:
                            writer.add_scalar('ep_r', score, global_step=total_steps)
                            writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                        print('EnvName:',EnvName[opt.EnvIdex],'Algo:',algo_name,'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))

                    '''save model'''
                    if total_steps % opt.save_interval == 0:
                        model.save(algo_name,EnvName[opt.EnvIdex],int(total_steps/1000))


    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()







