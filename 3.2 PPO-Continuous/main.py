from datetime import datetime
import os, shutil
import argparse
import torch
import gymnasium as gym

from utils import str2bool, Action_adapter, Reward_adapter, evaluate_policy
from PPO import PPO_agent


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
parser.add_argument('--Max_train_steps', type=int, default=int(5e7), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']

    # Build Env
    env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    opt.max_steps = env._max_episode_steps
    print('Env:',EnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,
          '  max_a:',opt.max_action,'  min_a:',env.action_space.low[0], 'max_steps', opt.max_steps)

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Use tensorboard to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Beta dist maybe need larger learning rate, Sometimes helps
    # if Dist[distnum] == 'Beta' :
    #     kwargs["a_lr"] *= 2
    #     kwargs["c_lr"] *= 4

    if not os.path.exists('model'): os.mkdir('model')
    agent = PPO_agent(**vars(opt)) # transfer opt to dictionary, and use it to init PPO_agent
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, opt.max_action, 1)
            print(f'Env:{EnvName[opt.EnvIdex]}, Episode Reward:{ep_r}')
    else:
        traj_lenth, total_steps = 0, 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                '''Interact with Env'''
                a, logprob_a = agent.select_action(s, deterministic=False) # use stochastic when training
                act = Action_adapter(a,opt.max_action) #[0,1] to [-max,max]
                s_next, r, dw, tr, info = env.step(act) # dw: dead&win; tr: truncated
                r = Reward_adapter(r, opt.EnvIdex)
                done = (dw or tr)

                '''Store the current transition'''
                agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
                s = s_next

                traj_lenth += 1
                total_steps += 1

                '''Update if its time'''
                if traj_lenth % opt.T_horizon == 0:
                    agent.train()
                    traj_lenth = 0

                '''Record & log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, opt.max_action, turns=3) # evaluate the policy for 3 times, and get averaged result
                    if opt.write: writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:',EnvName[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                '''Save model'''
                if total_steps % opt.save_interval==0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))

        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()





