import time
import copy
import torch
from utils import Q_Net
import torch.nn.functional as F
from copy import deepcopy
from utils import LinearSchedule


def learner_process(opt, shared_data):
	learner = Learner(opt, shared_data)
	learner.run()

class Learner:
	def __init__(self, opt, shared_data):
		self.shared_data = shared_data
		self.device = torch.device(opt.L_dvc)
		self.max_train_steps = opt.max_train_steps
		self.explore_steps = opt.explore_steps
		self.lr = opt.lr
		self.gamma = opt.gamma
		self.DDQN = opt.DDQN
		self.hard_update_freq = opt.hard_update_freq
		self.upload_freq = opt.upload_freq
		self.eval_freq = opt.eval_freq
		self.train_counter = 0
		self.batch_size = opt.batch_size

		self.q_net = Q_Net(opt.action_dim, opt.fc_width).to(self.device)
		self.upload_model()
		self.q_target = copy.deepcopy(self.q_net)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr, eps=1.5e-4)
		self.lr_scheduler = LinearSchedule(1.5e7, opt.lr, opt.lr/3)

		self.time_feedback = opt.time_feedback
		self.rho = opt.train_envs * opt.TPS / opt.batch_size
		# 使用tf时，一次vstep = rho次bpstep。
		# 因此，一次vstep的时间应该约等于rho次bpstep的时间
		# 当 t[vstep] < rho * t[bpstep]时，表明actor太快。这种情况下，每次vstep时，actor等待 (rho * t[bpstep] - t[vstep]) 秒
		# 当 t[vstep] > rho * t[bpstep]时，表明learner太快。这种情况下，每次bpstep时，learner等待 (t[vstep] - t[bpstep])/rho 秒

	def run(self):
		mean_t = 0 # average time of bp once
		while True:
			global_steps = self.shared_data.get_total_steps()
			if global_steps > self.max_train_steps: break #结束Learner进程

			if global_steps < self.explore_steps:
				time.sleep(0.1)
			else:
				t0 = time.time()
				self.train()
				self.train_counter += 1  # Bstep

				if self.train_counter % self.upload_freq == 0:
					self.upload_model()
					self.shared_data.set_should_download(True) #inform actor to download latest model

				if self.train_counter % self.hard_update_freq == 0:
					self.hard_target_update()
					self.lr_decay(global_steps)
					print('(Learner) Actual TPS: ',self.train_counter * self.batch_size / (global_steps-self.explore_steps))

				if self.train_counter % self.eval_freq == 0:
					self.shared_data.add_eval_model(deepcopy(self.q_net).cpu().state_dict(), global_steps-self.explore_steps, time.time()) #send model to evaluator

				if self.time_feedback:
					# 计算
					current_t = time.time() - t0 #本次训练消耗的时间
					mean_t = mean_t + (current_t-mean_t)/self.train_counter #增量法求得的平均训练时间
					# Object: 1 Vstep = rho * Bstep
					scalled_learner_time = self.rho * mean_t #
					# 存储
					self.shared_data.set_t(scalled_learner_time, 1) # learner时间放在第1位
					# 比较、等待
					t = self.shared_data.get_t()
					if t[1]<t[0]: #learner耗时少
						hold_time = (t[0]-t[1])/self.rho
						if hold_time > 1: hold_time = 1 #防止过长等待
						time.sleep( hold_time ) #learner等待(分成rho次等待)

	def train(self):
		s, a, r, s_next, dw, ct = self.shared_data.sample()

		'''Compute target Q value'''
		with torch.no_grad():
			if self.DDQN:
				argmax_a = self.q_net(s_next).argmax(dim=-1).unsqueeze(-1) #[b,1]
				max_q_next = self.q_target(s_next).gather(1,argmax_a) #[b,1]
			else:
				max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1)  #[b,1]

			target_Q = r + (~dw) * self.gamma * max_q_next  #[b,1]; dw: dead & win

		'''Collect Current Q value'''
		current_q = self.q_net(s)  # [b,a_dim]
		current_q_a = current_q.gather(1,a)

		#ct表示s和s_next是否来自一个回合，如果不是，则丢掉。
		if ct.all():
			q_loss = F.mse_loss(current_q_a, target_Q) #如果所有的都ct,则直接MSE
		else:
			q_loss = torch.square(ct * (current_q_a - target_Q)).mean() #否则要丢掉 不ct的数据
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 40)
		self.q_net_optimizer.step()

	def upload_model(self):
		# 好像不是很高效，如何优化?
		self.shared_data.set_net_param(deepcopy(self.q_net).cpu().state_dict())

	def hard_target_update(self):
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(param.data)
			target_param.requires_grad = False

	def lr_decay(self,global_step):
		for p in self.q_net_optimizer.param_groups:
			p['lr'] = self.lr_scheduler.value(global_step)


