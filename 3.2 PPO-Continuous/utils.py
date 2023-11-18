import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal


class BetaActor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(BetaActor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.alpha_head = nn.Linear(net_width, action_dim)
		self.beta_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))

		alpha = F.softplus(self.alpha_head(a)) + 1.0
		beta = F.softplus(self.beta_head(a)) + 1.0

		return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	def deterministic_act(self, state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode

class GaussianActor_musigma(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(GaussianActor_musigma, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.sigma_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		sigma = F.softplus( self.sigma_head(a) )
		return mu,sigma

	def get_dist(self, state):
		mu,sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

	def deterministic_act(self, state):
		mu, sigma = self.forward(state)
		return mu


class GaussianActor_mu(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(GaussianActor_mu, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		return mu

	def get_dist(self,state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist

	def deterministic_act(self, state):
		return self.forward(state)


class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v

def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		print('Wrong Input.')
		raise


def Action_adapter(a,max_action):
	#from [0,1] to [-max,max]
	return  2*(a-0.5)*max_action

def Reward_adapter(r, EnvIdex):
	# For BipedalWalker
	if EnvIdex == 0 or EnvIdex == 1:
		if r <= -100: r = -1
	# For Pendulum-v0
	elif EnvIdex == 3:
		r = (r + 8) / 8
	return r

def evaluate_policy(env, agent, max_action, turns):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
			act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
			s_next, r, dw, tr, info = env.step(act)
			done = (dw or tr)

			total_scores += r
			s = s_next

	return total_scores/turns