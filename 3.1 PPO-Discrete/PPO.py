from torch.distributions import Categorical
from utils import Actor, Critic
import numpy as np
import torch
import copy
import math


class PPO_discrete():
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        '''Build Actor and Critic'''
        self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        '''Build Trajectory holder'''
        self.s_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)

    def select_action(self, s, deterministic):
        s = torch.from_numpy(s).float().to(self.dvc)
        with torch.no_grad():
            pi = self.actor.pi(s, softmax_dim=0)
            if deterministic:
                a = torch.argmax(pi).item()
                return a, None
            else:
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[a].item()
                return a, pi_a

    def train(self):
        self.entropy_coef *= self.entropy_coef_decay #exploring decay
        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder).to(self.dvc)
        a = torch.from_numpy(self.a_hoder).to(self.dvc)
        r = torch.from_numpy(self.r_hoder).to(self.dvc)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
        done = torch.from_numpy(self.done_hoder).to(self.dvc)
        dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        for _ in range(self.K_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.dvc)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                '''actor update'''
                prob = self.actor.pi(s[index], softmax_dim=1)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/ppo_critic{}.pth".format(episode))
        torch.save(self.actor.state_dict(), "./model/ppo_actor{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./model/ppo_critic{}.pth".format(episode), map_location=self.dvc))
        self.actor.load_state_dict(torch.load("./model/ppo_actor{}.pth".format(episode), map_location=self.dvc))

