import numpy as np
import torch
from Agent import DeepQ_Agent, ReplayBuffer_torch
import os, shutil
from datetime import datetime
import argparse
from utils import evaluate_policy, str2bool, LinearSchedule
from tianshou_wrappers import make_env_tianshou
from AtariNames import Name

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='running device of algorithm: cuda or cpu')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=900, help='which model to load')

parser.add_argument('--Max_train_steps', type=int, default=int(1E6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1E5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(1e4), help='random steps before training, 5E4 in DQN Nature')
parser.add_argument('--buffersize', type=int, default=int(1e4), help='Size of the replay buffer')
parser.add_argument('--target_freq', type=int, default=int(1E3), help='frequency of target net updating')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--init_e', type=float, default=1.0, help='Initial e-greedy noise')
parser.add_argument('--anneal_frac', type=int, default=3e5, help='annealing fraction of e-greedy noise')
parser.add_argument('--final_e', type=float, default=0.02, help='Final e-greedy noise')
parser.add_argument('--noop_reset', type=str2bool, default=False, help='use NoopResetEnv or not')
parser.add_argument('--huber_loss', type=str2bool, default=True, help='True: use huber_loss; False:use mse_loss')
parser.add_argument('--fc_width', type=int, default=200, help='number of units in Fully Connected layer')

parser.add_argument('--EnvIdex', type=int, default=37, help='Index of the Env; 20=Enduro; 37=Pong')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--Double', type=str2bool, default=False, help="whether to use Double Q-learning")
parser.add_argument('--Duel', type=str2bool, default=False, help="whether to use Duel. Q-learning")
parser.add_argument('--Noisy', type=str2bool, default=False, help="whether to use NoisyNet")
opt = parser.parse_args()
opt.dvc = torch.device(opt.device)
opt.algo_name = ('Double-' if opt.Double else '') + ('Duel-' if opt.Duel else '') + ('Noisy-' if opt.Noisy else '') + 'DQN'
opt.EnvName = Name[opt.EnvIdex] + "NoFrameskip-v4"
opt.ExperimentName = opt.algo_name + '_' + opt.EnvName
print(opt)

def main():
    # Seed Everything
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build evaluation env
    render_mode = 'human' if opt.render else None
    eval_env = make_env_tianshou(opt.EnvName, noop_reset=opt.noop_reset, episode_life=False, clip_rewards=False, render_mode=render_mode)
    opt.action_dim = eval_env.action_space.n
    print('Algorithm:',opt.algo_name,'  Env:',opt.EnvName,'  Action_dim:',opt.action_dim,'  Seed:',opt.seed, '\n')

    #Build Agent
    if not os.path.exists('model'): os.mkdir('model')
    agent = DeepQ_Agent(opt)
    if opt.Loadmodel: agent.load(opt.ExperimentName,opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy(eval_env, agent, seed=opt.seed, turns=1)
            print(opt.ExperimentName, 'seed:', opt.seed, 'score:', score)
    else:
        if opt.write:
            from torch.utils.tensorboard import SummaryWriter
            timenow = str(datetime.now())[0:-7]
            timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2::]
            writepath = f'runs/{opt.ExperimentName}_S{opt.seed}' + timenow
            if os.path.exists(writepath): shutil.rmtree(writepath)
            writer = SummaryWriter(log_dir=writepath)

        # Build replay buffer and training env
        buffer = ReplayBuffer_torch(device=opt.dvc, max_size=opt.buffersize)
        env = make_env_tianshou(opt.EnvName, noop_reset = opt.noop_reset)

        #explore noise linearly annealed from 1.0 to 0.02 within anneal_frac steps
        schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=opt.final_e, initial_p=opt.init_e)
        agent.exp_noise = opt.init_e
        seed = opt.seed

        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=seed)
            seed += 1 # 每次reset都使用新的seed,防止overfitting
            done = False
            while not done:
                a = agent.select_action(s, evaluate=False)
                s_next, r, dw, tr, info = env.step(a) # dw(dead & win): terminated; tr: truncated
                buffer.add(s, a, r, s_next, dw)
                done = dw + tr
                s = s_next

                # train, e-decay, log, save
                if buffer.size >= opt.random_steps:
                    agent.train(buffer)

                    '''record & log'''
                    if total_steps % opt.eval_interval == 0:
                        score = evaluate_policy(eval_env, agent, seed=seed+1) # 不与当前训练seed重合，更general
                        if opt.write:
                            writer.add_scalar('ep_r', score, global_step=total_steps)
                            writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                        print(f"{opt.ExperimentName}, Seed:{opt.seed}, Step:{int(total_steps/1000)}k, Score:{score}")
                        agent.exp_noise = schedualer.value(total_steps) # e-greedy decay

                    total_steps += 1
                    '''save model'''
                    if total_steps % opt.save_interval == 0:
                        agent.save(opt.ExperimentName,int(total_steps/1000))

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()







