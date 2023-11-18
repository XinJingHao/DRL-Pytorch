import torch
import torch.nn as nn
from datetime import datetime


class Q_Net(nn.Module):
	def __init__(self, action_dim, hidden):
		super(Q_Net, self).__init__()
		self.net = nn.Sequential(
			# self.orthogonal_init(nn.Conv2d(4, 32, 8, stride=4)),
			nn.Conv2d(4, 32, 8, stride=4),
			nn.ReLU(),
			# self.orthogonal_init(nn.Conv2d(32, 64, 4, stride=2)),
			nn.Conv2d(32, 64, 4, stride=2),
			nn.ReLU(),
			# self.orthogonal_init(nn.Conv2d(64, 64, 3, stride=1)),
			nn.Conv2d(64, 64, 3, stride=1),
			nn.ReLU(),
			nn.Flatten(),
			# self.orthogonal_init(nn.Linear(64 * 7 * 7, hidden)),
			nn.Linear(64 * 7 * 7, hidden),
			nn.ReLU(),
			# self.orthogonal_init(nn.Linear(hidden, action_dim)),
			nn.Linear(hidden, action_dim)
		)

	def forward(self, obs):
		s = obs.float()/255 #convert to f32 and normalize before feeding to network
		q = self.net(s)
		return q

	def orthogonal_init(self, layer, gain=1.4142):
		for name, param in layer.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0)
			elif 'weight' in name:
				nn.init.orthogonal_(param, gain=gain)
		return layer

class LinearSchedule(object):
	def __init__(self, schedule_timesteps, initial_p, final_p):
		"""Linear interpolation between initial_p and final_p over
		schedule_timesteps. After this many timesteps pass final_p is
		returned.
		Parameters
		----------
		schedule_timesteps: int
			Number of timesteps for which to linearly anneal initial_p
			to final_p
		initial_p: float
			initial output value
		final_p: float
			final output value
		"""
		self.schedule_timesteps = schedule_timesteps
		self.initial_p = initial_p
		self.final_p = final_p

	def value(self, t):
		fraction = min(float(t) / self.schedule_timesteps, 1.0)
		return self.initial_p + fraction * (self.final_p - self.initial_p)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')