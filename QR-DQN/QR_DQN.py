import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QRDQNAgent:
    def __init__(self, s_dim, a_dim, lr, gamma, n_quantiles, kappa, exp_noise):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr = lr
        self.gamma = gamma
        self.n_quantiles = n_quantiles
        self.kappa = kappa
        self.exp_noise = exp_noise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = self.build_network().to(self.device)
        self.target_net = self.build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.update_target()

    def build_network(self):
        return nn.Sequential(
            nn.Linear(self.s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.a_dim * self.n_quantiles)
        )

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        quantiles = self.q_net(state).view(self.a_dim, self.n_quantiles)
        q_values = quantiles.mean(dim=1)
        if deterministic:
            action = q_values.argmax().item()
        else:
            if np.random.rand() < self.exp_noise:
                action = np.random.randint(0, self.a_dim)
            else:
                action = q_values.argmax().item()
        return action

    def train(self, s, a, r, s_next, dw):
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor([a]).to(self.device)
        r = torch.FloatTensor([r]).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        dw = torch.FloatTensor([dw]).to(self.device)

        quantiles = self.q_net(s).view(self.a_dim, self.n_quantiles)
        quantiles = quantiles[a].view(self.n_quantiles)

        with torch.no_grad():
            next_quantiles = self.target_net(s_next).view(self.a_dim, self.n_quantiles)
            next_q_values = next_quantiles.mean(dim=1)
            next_action = next_q_values.argmax().item()
            next_quantiles = next_quantiles[next_action].view(self.n_quantiles)
            target_quantiles = r + (1 - dw) * self.gamma * next_quantiles

        loss = self.quantile_regression_loss(quantiles, target_quantiles)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def quantile_regression_loss(self, quantiles, target_quantiles):
        diff = target_quantiles.unsqueeze(0) - quantiles.unsqueeze(1)
        loss = torch.where(
            diff < 0,
            (1 - self.kappa) * diff ** 2,
            self.kappa * diff ** 2
        )
        return loss.mean()

    def save(self):
        torch.save(self.q_net.state_dict(), 'model/qr_dqn.pth')

    def restore(self):
        self.q_net.load_state_dict(torch.load('model/qr_dqn.pth'))
        self.update_target()

def evaluate_policy(env, agent, turns=3):
    total_reward = 0
    for _ in range(turns):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
    return total_reward / turns
