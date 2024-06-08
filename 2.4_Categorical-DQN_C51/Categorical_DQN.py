import torch.nn as nn
import numpy as np
import torch
import copy


def build_net(layer_shape, activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

class Categorical_Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, atoms):
		super(Categorical_Q_Net, self).__init__()
		self.atoms = atoms
		self.n_atoms = len(atoms)
		self.action_dim = action_dim

		layers = [state_dim] + list(hid_shape) + [action_dim*self.n_atoms]
		self.net = build_net(layers, nn.ReLU, nn.Identity)

	def _predict(self, state):
		logits = self.net(state) # (batch_size, action_dim*n_atoms)
		distributions = torch.softmax(logits.view(len(state), self.action_dim, self.n_atoms), dim=2) # (batch_size, a_dim, n_atoms)
		q_values = (distributions * self.atoms).sum(2) # (batch_size, a_dim)
		return distributions, q_values

	def forward(self, state, action=None):
		distributions, q_values = self._predict(state)
		if action is None:
			action = torch.argmax(q_values, dim=1)
		return action, distributions[torch.arange(len(state)), action]




class CDQN_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.atoms = torch.linspace(self.v_min, self.v_max, steps=self.n_atoms,device=self.dvc)
		self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
		self.m = torch.zeros((self.batch_size, self.n_atoms), device=self.dvc)

		self.q_net = Categorical_Q_Net(self.state_dim, self.action_dim, (self.net_width,self.net_width),self.atoms).to(self.dvc)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False

		self.offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size,device=self.dvc).unsqueeze(-1).long()
		self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))
		self.tau = 0.005

	def select_action(self, state, deterministic):
		# Only be used when interacting with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if (not deterministic) and (np.random.rand() < self.exp_noise):
				return np.random.randint(0,self.action_dim)
			else:
				a, _ = self.q_net(state)
				return a.cpu().item()


	def train(self):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size) # dw(terminate): die or win

		'''Compute the target distribution:'''
		with torch.no_grad():
			# Note that the original paper just use Single Q-learning, but we find Double Q-learning more stable.
			if self.DQL:
				argmax_a_next, _ = self.q_net(s_next) # (batch_size,)
				_, batched_next_distribution = self.q_target(s_next, argmax_a_next) # _, (batch_size, n_atoms)  # Double Q-learning
			else:
				_, batched_next_distribution = self.q_target(s_next)  # _, (batch_size, n_atoms)  # Single Q-learning

			self.m *= 0
			t_z = (r + (~dw) * self.gamma * self.atoms).clamp(self.v_min, self.v_max) # (batch_size, n_atoms)
			b = (t_z - self.v_min)/self.delta_z # b∈[0,n_atoms-1]; shape: (batch_size, n_atoms)
			l = b.floor().long()  # (batch_size, n_atoms)
			u = b.ceil().long()  # (batch_size, n_atoms)

			# When bj is exactly an integer, then bj.floor() == bj.ceil(), then u should +1.
			# Eg: bj=1, l=1, u should = 2
			delta_m_l = (u + (l == u) - b) * batched_next_distribution  # (batch_size, n_atoms)
			delta_m_u = (b - l) * batched_next_distribution # (batch_size, n_atoms)


			'''Distribute probability with tensor operation. Much more faster than the For loop in the original paper.'''
			self.m.view(-1).index_add_(0, (l + self.offset).view(-1), delta_m_l.view(-1))
			self.m.view(-1).index_add_(0, (u + self.offset).view(-1), delta_m_u.view(-1))

		# Get current estimate:
		_, batched_distribution = self.q_net(s, a.flatten()) # _, (batch_size, n_atoms)

		# Compute Corss Entropy Loss:
		# q_loss = (-(self.m * batched_distribution.log()).sum(-1)).mean() # Original Cross Entropy loss, not stable
		q_loss = (-(self.m * batched_distribution.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean() # more stable

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




