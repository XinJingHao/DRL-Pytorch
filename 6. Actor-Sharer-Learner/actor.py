import time
import envpool
import torch
import numpy as np
from utils import Q_Net, LinearSchedule

def actor_process(opt, shared_data):
	actor = Actor(opt, shared_data)
	actor.run()

class Actor:
	def __init__(self, opt, shared_data):
		# Basic information init
		self.shared_data = shared_data
		self.device = torch.device(opt.A_dvc)
		self.max_train_steps = opt.max_train_steps
		self.train_envs = opt.train_envs
		self.action_dim = opt.action_dim
		self.explore_steps = opt.explore_steps
		self.time_feedback = opt.time_feedback

		# vectorized e-greedy exploration mechanism
		self.explore_frac_scheduler = LinearSchedule(opt.decay_step, opt.init_explore_frac, opt.end_explore_frac)
		self.p = torch.zeros(opt.train_envs)
		self.min_eps = opt.min_eps

		# build actor and envs
		# 参考Apex训练时的最大步长为50e3 frame, envpool中最大步长按step算的，不是frame，所以还要除4
		self.envs = envpool.make_gym(opt.ExpEnvName, num_envs=opt.train_envs, seed=opt.seed,
									 max_episode_steps=int(50e3 / 4), episodic_life=True, reward_clip=True)
		self.actor_net = Q_Net(opt.action_dim, opt.fc_width).to(self.device)
		# self.download_model()
		self.step_counter = 0 #local step counter in Actor

	def run(self):
		ct = np.ones(self.train_envs,dtype=np.bool_) # consistent(Check AutoReset in Envpool)
		s, info = self.envs.reset()
		mean_t, c = 0, 0
		while True:
			if self.step_counter > self.max_train_steps: break #结束Actor进程

			random_phase = self.step_counter < self.explore_steps
			if random_phase:
				a = np.random.randint(0, self.action_dim, self.train_envs)
			else:
				t0 = time.time()
				a = self.select_action(s)
			s_next, r, dw, tr, info = self.envs.step(a)
			self.shared_data.add(s, a, r, dw, ct) #注意ct是用上一次step的， 即buffer.add()要在ct = ~(dw + tr)前
			ct = ~(dw + tr)  # 如果当前s_next是”截断状态“或”终止状态“，则s_next与s_next_next是不consistent的，训练时要丢掉
			s = s_next

			#update gloabel steps
			self.step_counter += self.train_envs
			self.shared_data.set_total_steps(self.step_counter)

			if not random_phase:
				# download model parameters from shared_data.net_param
				if self.step_counter % (5*self.train_envs) == 0: # don't ask shared_data too frequently
					if self.shared_data.get_should_download():
						self.shared_data.set_should_download(False)
						self.download_model()

				# fresh vectorized e-greedy noise
				if self.step_counter % (10*self.train_envs) == 0:
					self.fresh_explore_prob(self.step_counter-self.explore_steps)

				if self.step_counter % (100 * self.train_envs) == 0:
					print('(Actor) Tstep: {}k'.format(int(self.step_counter/1e3)) )

				if self.time_feedback:
					# 计算
					c += 1
					current_t = time.time() - t0  # 本次step消耗的时间
					mean_t = mean_t + (current_t - mean_t) / c  # 增量法求得的平均step时间
					# 存储
					self.shared_data.set_t(mean_t, 0) # actor时间放在第0位
					# 比较、等待
					t = self.shared_data.get_t()
					if t[0]<t[1]: #actor耗时少
						hold_time = t[1]-t[0]
						if hold_time > 1: hold_time = 1 #防止过长等待
						time.sleep(hold_time) #actor等待

	def fresh_explore_prob(self, steps):
		#fresh vectorized e-greedy noise
		explore_frac = self.explore_frac_scheduler.value(steps)  # 1.0 -> 0.032 in decay_step
		i = int(explore_frac * self.train_envs) # 128 -> 4
		explore = torch.arange(i) / (1.25 * i)  # [0,0.8]
		self.p *= 0
		self.p[self.train_envs - i:] = explore
		self.p += self.min_eps


	def select_action(self, s):
		'''For envpool, the input is [n,4,84,84], npdarray'''
		with torch.no_grad():
			s = torch.from_numpy(s).to(self.device)  # [b,s_dim]
			a = self.actor_net(s).argmax(dim=-1).cpu()  # [b]
			replace = torch.rand(self.train_envs) < self.p  # [b]
			rd_a = torch.randint(0, self.action_dim, (self.train_envs,))
			a[replace] = rd_a[replace]
			return a.numpy()

	def download_model(self):
		self.actor_net.load_state_dict(self.shared_data.get_net_param())
		for actor_param in self.actor_net.parameters():
			actor_param.requires_grad = False