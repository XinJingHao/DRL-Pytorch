from collections import deque
from utils import Q_Net
import numpy as np
import envpool
import torch
import time

def eval_process(eid, opt, shared_data):
	evaluator = Evaluator(eid, opt, shared_data)
	evaluator.run()

class Evaluator:
	def __init__(self, eid, opt, shared_data):
		self.eid = eid
		self.shared_data = shared_data
		self.device = torch.device(opt.E_dvc)
		self.envname = opt.ExpEnvName
		self.eval_envs = opt.eval_envs
		self.max_train_steps = opt.max_train_steps

		# build actor and envs
		self.eval_net = Q_Net(opt.action_dim, opt.fc_width).to(self.device)
		self.envs = envpool.make_gym(self.envname, num_envs=opt.eval_envs, seed=opt.seed + 1,
									 max_episode_steps=int(108e3 / 4), episodic_life=False, reward_clip=False)
		# Because of the reset() bug in episodic_life wrapper, episodic_life should be False in evaluator
		# Reset() Bug: when was_real_done=False, reset() = step(0). Thus, if episodic_life=True, You can't do real reset whenever you want.

	def run(self):
		while True:
			data = self.shared_data.get_eval_model() #{'model': params, 'steps': global_steps, 'time':walltime}

			global_steps = self.shared_data.get_total_steps() # 这里仅用于结束evaluator进程，不用于画图
			if (global_steps > self.max_train_steps) and (data is None): break #结束Evaluator进程

			if data is None:
				time.sleep(5)
			else:
				self.eval_net.load_state_dict(data['model'])
				for eval_param in self.eval_net.parameters():
					eval_param.requires_grad = False
				score = self.evaluate()

				self.shared_data.add_curvepoint([score, data['steps'], data['time']]) # 存入curve上的一个点，后面Recorder统一画
				print('(Evaluator {}) '.format(self.eid),self.envname,'  Tstep:{}k'.format(round(data['steps'] / 1000, 2)),'  score:', score)


	def evaluate(self):
		s, info = self.envs.reset()
		dones, total_r = np.zeros(self.eval_envs, dtype=np.bool_), 0
		while not dones.all():
			a = self.select_action(s)
			s, r, dw, tr, info = self.envs.step(a)
			total_r += (~dones * r).sum()  # use last dones
			dones += (dw + tr) # use current dones
		return round(total_r / self.eval_envs, 1)


	def select_action(self, s):
		'''for envpool'''
		with torch.no_grad():
			s = torch.from_numpy(s).to(self.device)  # [b,s_dim]
			return self.eval_net(s).argmax(dim=-1).cpu().numpy()  # [b]


