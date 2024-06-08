import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from utils import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_net(layer_shape, activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.Q = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q = self.Q(s)
		return q



class DQN_Agent(object):
	def __init__(self,opt,):
		self.q_net = Q_Net(opt.state_dim, opt.action_dim, (opt.net_width,opt.net_width)).to(device)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.gamma = opt.gamma
		self.tau = 0.005
		self.batch_size = opt.batch_size
		self.exp_noise = opt.exp_noise_init
		self.action_dim = opt.action_dim
		self.DDQN = opt.DDQN

	def select_action(self, state, deterministic):#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			if deterministic:
				a = self.q_net(state).argmax().item()
			else:
				if np.random.rand() < self.exp_noise:
					a = np.random.randint(0,self.action_dim)
				else:
					a = self.q_net(state).argmax().item()
		return a


	def train(self,replay_buffer):
		s, a, r, s_prime, dw_mask, ind, Normed_IS_weight = replay_buffer.sample(self.batch_size)

		'''Compute the target Q value'''
		with torch.no_grad():
			if self.DDQN:
				argmax_a = self.q_net(s_prime).argmax(dim=1).unsqueeze(-1)
				max_q_prime = self.q_target(s_prime).gather(1,argmax_a)
			else:
				max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)

			'''Avoid impacts caused by reaching max episode steps'''
			target_Q = r + (1 - dw_mask) * self.gamma * max_q_prime #dw: die or win; shape：(batch_size, 1)

		# Get current Q estimates
		current_q_a = self.q_net(s).gather(1,a)  # shape：(batch_size, 1)

		td_errors = (current_q_a - target_Q).squeeze(-1) # shape：(batch_size,)
		loss = (Normed_IS_weight * (td_errors ** 2)).mean()

		replay_buffer.update_batch_priorities(ind, td_errors.detach().cpu().numpy())

		self.q_net_optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
		self.q_net_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self,algo,EnvName,steps):
		torch.save(self.q_net.state_dict(), "./model/{}_{}_{}.pth".format(algo,EnvName,steps))

	def load(self,algo,EnvName,steps):
		self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps), map_location=device))
		self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps), map_location=device))


class PrioritizedReplayBuffer(object):
	def __init__(self, opt):

		self.ptr = 0
		self.size = 0

		max_size = int(opt.buffer_size)
		self.state = np.zeros((max_size, opt.state_dim))
		self.action = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, opt.state_dim))
		self.dw = np.zeros((max_size, 1))
		self.max_size = max_size

		self.sum_tree = SumTree(max_size)
		self.alpha = opt.alpha
		self.beta = opt.beta_init

		self.device = device


	def add(self, state, action, reward, next_state, dw):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dw[self.ptr] = dw #0,0,0，...，1

		# 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
		priority = 1.0 if self.size == 0 else self.sum_tree.priority_max
		self.sum_tree.update_priority(data_index=self.ptr, priority=priority)  # 更新当前经验在sum_tree中的优先级

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind, Normed_IS_weight = self.sum_tree.prioritized_sample(N=self.size, batch_size=batch_size, beta=self.beta)

		return (
			torch.tensor(self.state[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.action[ind], dtype=torch.long).to(self.device),
			torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.next_state[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.dw[ind], dtype=torch.float32).to(self.device),
			ind,
			Normed_IS_weight.to(self.device) # shape：(batch_size,)
		)

	def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
		priorities = (np.abs(td_errors) + 0.01) ** self.alpha
		for index, priority in zip(batch_index, priorities):
			self.sum_tree.update_priority(data_index=index, priority=priority)







