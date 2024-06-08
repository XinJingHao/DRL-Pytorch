from utils import Actor, Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class DDPG_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), dvc=self.dvc)
		
	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
				return (a + noise).clip(-self.max_action, self.max_action)

	def train(self):
		# Compute the target Q
		with torch.no_grad():
			s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
			target_a_next = self.actor_target(s_next)
			target_Q= self.q_critic_target(s_next, target_a_next)
			target_Q = r + (~dw) * self.gamma * target_Q  #dw: die or win

		# Get current Q estimates
		current_Q = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		# Update the Actor
		a_loss = -self.q_critic(s,self.actor(s)).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		with torch.no_grad():
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]



