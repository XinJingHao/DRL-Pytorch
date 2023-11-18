import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LightPriorReplayBuffer():
    '''
    Obviate the need for explicately saving s_next, more menmory friendly, especially for image state.

    When iterating, use the following way to add new transitions:
        a = model.select(s)
        s_next, r, dw, tr, info = env.step(a)
        buffer.add(s, a, r, dw, tr)  
        # dw: whether the 's_next' is the terminal state
        # tr: whether the episode has been truncated.

    When sampling,
    ind = [ptr - 1] and ind = [size - 1] should be avoided to ensure the consistence of state[ind] and state[ind+1]
    Then,
    s = self.state[ind]
    s_next = self.state[ind+1]

    Importantly, because we do not explicitly save 's_next', when dw or tr is True, the s[ind] and s[ind+1] is not from one episode. 
    when encounter dw=True,
    self.state[ind+1] is not the true next state of self.state[ind], but a new resetted state.
    It doesn't matter, since Q_target[s[ind],a[ind]] = r[ind] + gamma*(1-dw[ind])* max_Q(s[ind+1],·),
    when dw=true, we won't use s[ind+1] at all.
    however, when encounter tr=True,
    self.state[ind+1] is not the true next state of self.state[ind], but a new resetted state, 
    so we have to discard this transition through (1-tr) in the loss function

    Thus, when training,
    Q_target = r + self.gamma * (1-dw) * max_q_next
    current_Q = self.q_net(s).gather(1,a)
    q_loss = torch.square((1-tr) * (current_Q - Q_target)).mean()

    '''

    def __init__(self, opt):
        self.device = device
        
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((opt.buffer_size, opt.state_dim), device=device)  #如果是图像，可以用unit8节省空间
        self.action = torch.zeros((opt.buffer_size, 1), dtype=torch.int64, device=device)
        self.reward = torch.zeros((opt.buffer_size, 1), device=device)
        self.dw = torch.zeros((opt.buffer_size, 1), dtype=torch.bool, device=device) #only 0/1
        self.tr = torch.zeros((opt.buffer_size, 1), dtype=torch.bool, device=device) #only 0/1
        self.priorities = torch.zeros(opt.buffer_size, dtype=torch.float32, device=device) # (|TD-error|+0.01)^alpha
        self.buffer_size = opt.buffer_size

        self.alpha = opt.alpha
        self.beta = opt.beta_init
        self.replacement = opt.replacement


    def add(self, state, action, reward, dw, tr, priority):
        self.state[self.ptr] = torch.from_numpy(state).to(device)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.dw[self.ptr] = dw
        self.tr[self.ptr] = tr
        self.priorities[self.ptr] = priority

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)


    def sample(self, batch_size):
        # 因为state[ptr-1]和state[ptr]，state[size-1]和state[size]不来自同一个episode
        Prob_torch_gpu = self.priorities[0: self.size - 1].clone() # 所以从[0, size-1)中sample; 这里必须clone
        if self.ptr < self.size: Prob_torch_gpu[self.ptr-1] = 0 # 并且不能包含ptr-1
        ind = torch.multinomial(Prob_torch_gpu, num_samples=batch_size, replacement=self.replacement) # replacement=True数据可能重复，但是快很多; (batchsize,)
        # 注意，这里的ind对于self.priorities和Prob_torch_gpu是通用的，并没有错位

        IS_weight = (self.size * Prob_torch_gpu[ind])**(-self.beta)
        Normed_IS_weight = (IS_weight / IS_weight.max()).unsqueeze(-1) #(batchsize,1)

        return self.state[ind], self.action[ind], self.reward[ind], self.state[ind + 1], self.dw[ind], self.tr[ind], ind, Normed_IS_weight
