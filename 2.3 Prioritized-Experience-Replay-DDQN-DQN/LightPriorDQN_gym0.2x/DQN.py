import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from LPRB import device
import math


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
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr_init)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.env_with_dw = opt.env_with_dw
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
				return a
			else:
				Q = self.q_net(state)
				if np.random.rand() < self.exp_noise:
					a = np.random.randint(0,self.action_dim)
					q_a = Q[0,a] # on device
				else:
					a = Q.argmax().item()
					q_a = Q[0,a] # on device
				return a, q_a


	def train(self,replay_buffer):
		s, a, r, s_next, dw, tr, ind, Normed_IS_weight = replay_buffer.sample(self.batch_size)
		# s, a, r, s_next, dw, tr, Normed_IS_weight : (batchsize, dim)
		# ind, : (batchsize,)

		'''Compute the target Q value'''
		with torch.no_grad():
			if self.DDQN:
				argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
				max_q_prime = self.q_target(s_next).gather(1,argmax_a)
			else:
				max_q_prime = self.q_target(s_next).max(1)[0].unsqueeze(1)

			'''Avoid impacts caused by reaching max episode steps'''
			Q_target = r + (~dw) * self.gamma * max_q_prime #dw: die or win

		# Get current Q estimates
		current_Q = self.q_net(s).gather(1,a)

		# BP
		q_loss = torch.square((~tr) * Normed_IS_weight * (Q_target - current_Q)).mean()
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
		self.q_net_optimizer.step()

		# update priorites of the current batch
		with torch.no_grad():
			batch_priorities = ((torch.abs(Q_target - current_Q) + 0.01)**replay_buffer.alpha).squeeze(-1) #(batchsize,) on devive
			replay_buffer.priorities[ind] = batch_priorities

		# Update the frozen target models
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self,algo,EnvName,steps):
		torch.save(self.q_net.state_dict(), "./model/{}_{}_{}.pth".format(algo,EnvName,steps))

	def load(self,algo,EnvName,steps):
		self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps), map_location=device))
		self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps), map_location=device))









