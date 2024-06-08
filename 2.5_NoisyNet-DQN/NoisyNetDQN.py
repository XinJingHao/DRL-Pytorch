import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
from utils import NoisyLinear


def build_net(layer_shape, activation, output_activation):
	'''Build networks with For loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		if j < len(layer_shape) - 2: layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
		else: layers += [NoisyLinear(layer_shape[j], layer_shape[j+1], sigma_init=0.25), output_activation()]
	return nn.Sequential(*layers)

class Noisy_Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Noisy_Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.Q = build_net(layers, nn.ReLU, nn.Identity)
	def forward(self, s):
		q = self.Q(s)
		return q



class NoisyNetDQN_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005
		self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=self.buffer_size)
		self.q_net = Noisy_Q_Net(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False


	def select_action(self, state):
		# NoisyNet无需e-greedy
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			a = self.q_net(state).argmax().item()
		return a


	def train(self):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		'''Compute the target Q value'''
		with torch.no_grad():
			max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1)
			target_Q = r + (~dw) * self.gamma * max_q_next #dw: die or win

		# Get current Q estimates
		current_q = self.q_net(s)
		current_q_a = current_q.gather(1,a)

		q_loss = F.mse_loss(current_q_a, target_Q)
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		self.q_net_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self,algo,EnvName,steps):
		torch.save(self.q_net.state_dict(), "./model/{}_{}_{}k.pth".format(algo,EnvName,steps))

	def load(self,algo,EnvName,steps):
		self.q_net.load_state_dict(torch.load("./model/{}_{}_{}k.pth".format(algo,EnvName,steps), map_location=self.dvc))
		self.q_target.load_state_dict(torch.load("./model/{}_{}_{}k.pth".format(algo,EnvName,steps), map_location=self.dvc))


class ReplayBuffer(object):
	def __init__(self, state_dim, dvc, max_size=int(1e6)):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.dvc)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = a
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]




