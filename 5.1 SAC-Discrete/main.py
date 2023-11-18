from utils import evaluate_policy, str2bool
from datetime import datetime
from SACD import SACD_agent
import gymnasium as gym
import os, shutil
import argparse
import torch

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=50, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=4e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hid_shape', type=list, default=[200,200], help='Hidden net shape')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
opt = parser.parse_args()
print(opt)

def main():
	#Create Env
	EnvName = ['CartPole-v1', 'LunarLander-v2']
	BriefEnvName = ['CPV1', 'LLdV2']
	env = gym.make(EnvName[opt.EnvIdex], render_mode="human" if opt.render else None)
	eval_env = gym.make(EnvName[opt.EnvIdex])
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.n
	opt.max_e_steps = env._max_episode_steps

	# Seed Everything
	env_seed = opt.seed
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	print("Random Seed: {}".format(opt.seed))

	print('Algorithm: SACD','  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
		  '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

	if opt.write:
		from torch.utils.tensorboard import SummaryWriter
		timenow = str(datetime.now())[0:-10]
		timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
		writepath = 'runs/SACD_{}'.format(BriefEnvName[opt.EnvIdex]) + timenow
		if os.path.exists(writepath): shutil.rmtree(writepath)
		writer = SummaryWriter(log_dir=writepath)

	#Build model
	if not os.path.exists('model'): os.mkdir('model')
	agent = SACD_agent(**vars(opt))
	if opt.Loadmodel: agent.load(opt.ModelIdex, BriefEnvName[opt.EnvIdex])

	if opt.render:
		while True:
			score = evaluate_policy(env, agent, 1)
			print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
	else:
		total_steps = 0
		while total_steps < opt.Max_train_steps:
			s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
			env_seed += 1
			done = False

			'''Interact & trian'''
			while not done:
				#e-greedy exploration
				if total_steps < opt.random_steps: a = env.action_space.sample()
				else: a = agent.select_action(s, deterministic=False)
				s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
				done = (dw or tr)

				if opt.EnvIdex == 1:
					if r <= -100: r = -10  # good for LunarLander

				agent.replay_buffer.add(s, a, r, s_next, dw)
				s = s_next

				'''update if its time'''
				# train 50 times every 50 steps rather than 1 training per step. Better!
				if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
					for j in range(opt.update_every):
						agent.train()

				'''record & log'''
				if total_steps % opt.eval_interval == 0:
					score = evaluate_policy(eval_env, agent, turns=3)
					if opt.write:
						writer.add_scalar('ep_r', score, global_step=total_steps)
						writer.add_scalar('alpha', agent.alpha, global_step=total_steps)
						writer.add_scalar('H_mean', agent.H_mean, global_step=total_steps)
					print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed,
						  'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
				total_steps += 1

				'''save model'''
				if total_steps % opt.save_interval == 0:
					agent.save(int(total_steps/1000), BriefEnvName[opt.EnvIdex])
	env.close()
	eval_env.close()


if __name__ == '__main__':
	main()

