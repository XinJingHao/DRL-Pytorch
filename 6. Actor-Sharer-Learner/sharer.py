import copy
import torch
import numpy as np
import time
from collections import deque


class shared_data_cpu():
	'''Using RAM to store expriences'''
	def __init__(self, opt):
		self.B_dvc = torch.device(opt.B_dvc) # buffer device
		self.L_dvc = torch.device(opt.L_dvc) # leaner device
		self.max_size = int(opt.buffersize/opt.train_envs)
		self.train_envs = opt.train_envs
		self.ptr = 0
		self.size = 0
		self.full = False
		self.batch_size = opt.batch_size

		# init shared data
		self.t = [0,0] # time feedback, 0是actor时间，1是scalled的learner时间
		self.net_param = None # net.state_dict(), upload/download model
		self.total_steps = 0
		self.eval_deque = deque() # deque storing model to be evaluate
		self.train_curve = [] # record the unsorted train curve points, which will be sorted in Recorder process
		self.should_download = False  # whether actor should download model

		# init shared buffer
		self.s = torch.zeros((self.max_size, opt.train_envs, 4, 84, 84), dtype=torch.uint8, device=self.B_dvc)
		self.a = torch.zeros((self.max_size, opt.train_envs, 1), dtype=torch.int64, device=self.B_dvc)
		self.r = torch.zeros((self.max_size, opt.train_envs, 1), device=self.B_dvc)
		self.dw = torch.zeros((self.max_size, opt.train_envs, 1), dtype=torch.bool, device=self.B_dvc)
		self.ct = torch.zeros((self.max_size, opt.train_envs, 1), dtype=torch.bool, device=self.B_dvc)  # mark if s[ind] and s[ind+1] belong to the same traj.


		# Tread lock (Naive Version)
		self.get_lock_time = 2e-4
		self.set_lock_time = 1e-4
		self.busy = [False, False, False] #标记某个共享数据是否正在被占用, F表示空闲，T表示正在get()/set()
		# self.busy[0] 标记net_param
		# self.busy[1] 标记buffer
		# self.busy[2] 标记train_curve

	def add(self, s, a, r, dw, ct):
		'''add transitions to buffer,with thread lock'''
		self.set_lock(self.add_core, 1, (s, a, r, dw, ct))  # use self.busy[1] to lock buffer data

	def add_core(self, trans):
		'''add transitions to buffer,without thread lock'''
		s, a, r, dw, ct = trans#[train_envs, s_dim]; [train_envs], npdarray
		self.s[self.ptr] = torch.from_numpy(s)
		self.a[self.ptr] = torch.from_numpy(a).unsqueeze(-1)
		self.r[self.ptr] = torch.from_numpy(r).unsqueeze(-1)
		self.dw[self.ptr] = torch.from_numpy(dw).unsqueeze(-1) # whether !next! s is dw
		self.ct[self.ptr] = torch.from_numpy(ct).unsqueeze(-1) # whether !next! s is consistent

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		if self.size == self.max_size:
			self.full = True

	def sample(self):
		'''sample batch transitions, with threading lock'''
		return self.get_lock(self.sample_core, 1)  # use self.busy[1] to lock buffer data

	def sample_core(self):
		'''sample batch transitions, without thread lock'''
		if not self.full:
			ind = torch.randint(low=0, high=self.ptr - 1, size=(self.batch_size,))  # sample from [0, ptr-2]
		else:
			ind = torch.randint(low=0, high=self.size - 1, size=(self.batch_size,))  # sample from [0, size-2]
			if self.ptr - 1 in ind: ind = ind[ind != (self.ptr - 1)]  # delate ptr - 1 in [0, size-2]
			#if self.ptr - 1 in ind: ind = np.delete(ind, np.where(ind == (self.ptr - 1)))  # delate ptr - 1 in [0, size-2]

		env_ind = torch.randint(low=0, high=self.train_envs, size=(len(ind),)) # [l,h)
		# [b, s_dim], #[b, 1], [b, 1], [b, s_dim], [b, 1], [b, 1]
		return (self.s[ind,env_ind,:].to(self.L_dvc), self.a[ind,env_ind,:].to(self.L_dvc),
				self.r[ind,env_ind,:].to(self.L_dvc), self.s[ind + 1,env_ind,:].to(self.L_dvc),
				self.dw[ind,env_ind,:].to(self.L_dvc), self.ct[ind, env_ind,:].to(self.L_dvc))


	def get_net_param(self):
		return self.get_lock(self.get_net_param_core, 0) # use self.busy[0] to lock net_param
	def get_net_param_core(self):
		return self.net_param

	def set_net_param(self, net_param):
		self.set_lock(self.set_net_param_core, 0, net_param) # use self.busy[0] to lock net_param
	def set_net_param_core(self, net_param):
		self.net_param = net_param

	def add_curvepoint(self, curvepoint):
		self.set_lock(self.add_curvepoint_core, 2, curvepoint)  # use self.busy[2] to lock train_curve
	def add_curvepoint_core(self, curvepoint):
		self.train_curve.append(curvepoint)  # curve_point = [value, step, walltime]

	def get_curve(self):
		return self.get_lock(self.get_curve_core, 2)  # use self.busy[2] to lock train_curve
	def get_curve_core(self):
		curve = copy.deepcopy(self.train_curve)
		self.train_curve = [] # 清空
		return curve

	#---------------------------------下面是没加进程锁的函数----------------------------#

	def add_eval_model(self, params, global_steps ,walltime):
		# add model needs evaluate to deque (learner)
		self.eval_deque.append({'model': params, 'steps': global_steps, 'time':walltime})
	def get_eval_model(self):
		# get model from deque and evaluate (evaluater)
		if self.eval_deque: # if there is data in deque
			return self.eval_deque.popleft()
		else:
			return None

	# Time feedback
	def get_t(self):
		return self.t
	def set_t(self,time,idx):
		self.t[idx] = time

	# 总交互次数
	def get_total_steps(self):
		return self.total_steps
	def set_total_steps(self,total_steps):
		self.total_steps = total_steps

	# Actor是否应该下载模型
	def get_should_download(self):
		return self.should_download
	def set_should_download(self, bol):
		self.should_download = bol


	# 锁函数
	def get_lock(self, get_func, idx):
		''' get_func is the function to be lock, idx is the index of self.busy '''
		while True:
			if self.busy[idx]:
				time.sleep(self.get_lock_time) #等待
			else:
				time.sleep(self.get_lock_time)  #Double check,防止同时占用 (get/set have different double check freq)
				if not self.busy[idx]:
					self.busy[idx] = True # 占用
					data = get_func() #被锁的操作
					self.busy[idx] = False  # 解除占用
					return data

	def set_lock(self, set_func, idx, data):
		''' set_func is the function to be lock, idx is the index of self.busy, data is the data to be set '''
		while True:
			if self.busy[idx]:
				time.sleep(self.set_lock_time)  # 等待
			else:
				time.sleep(self.set_lock_time)  # 以get()不同的频率再次check,防止同时占用
				if not self.busy[idx]:
					self.busy[idx] = True # 占用
					set_func(data) #被锁的操作
					self.busy[idx] = False # 解除占用
					break


'''---------------------------------------------------------------------------------------------------------------'''

class shared_data_cuda(shared_data_cpu):
	'''Using Cuda to store expriences'''
	def __init__(self, opt):
		super(shared_data_cuda, self).__init__(opt)

	def add_core(self, trans):
		'''add transitions to buffer,without thread lock'''
		s, a, r, dw, ct = trans#[train_envs, s_dim]; [train_envs], npdarray
		self.s[self.ptr] = torch.from_numpy(s).to(self.B_dvc)
		self.a[self.ptr] = torch.from_numpy(a).unsqueeze(-1).to(self.B_dvc)
		self.r[self.ptr] = torch.from_numpy(r).unsqueeze(-1).to(self.B_dvc)
		self.dw[self.ptr] = torch.from_numpy(dw).unsqueeze(-1).to(self.B_dvc) # whether !next! s is dw
		self.ct[self.ptr] = torch.from_numpy(ct).unsqueeze(-1).to(self.B_dvc) # whether !next! s is consistent

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		if self.size == self.max_size:
			self.full = True

	def sample_core(self):
		'''sample batch transitions, without thread lock'''
		if not self.full:
			ind = torch.randint(low=0, high=self.ptr - 1, size=(self.batch_size,), device=self.B_dvc)  # sample from [0, ptr-2]
		else:
			ind = torch.randint(low=0, high=self.size - 1, size=(self.batch_size,), device=self.B_dvc)  # sample from [0, size-2]
			if self.ptr - 1 in ind: ind = ind[ind != (self.ptr - 1)]  # delate ptr - 1 in [0, size-2]
			#if self.ptr - 1 in ind:ind = np.delete(ind, np.where(ind == (self.ptr - 1)))  # delate ptr - 1 in [0, size-2]

		env_ind = torch.randint(low=0, high=self.train_envs, size=(len(ind),), device=self.B_dvc) # [l,h)
		# [b, s_dim], #[b, 1], [b, 1], [b, s_dim], [b, 1], [b, 1]; 已经在L_dvc上了，因为使用GPUbuffer时，B_dvc=L_dvc
		return (self.s[ind,env_ind,:], self.a[ind,env_ind,:], self.r[ind,env_ind,:],
				self.s[ind + 1,env_ind,:], self.dw[ind,env_ind,:], self.ct[ind, env_ind,:])
