import numpy as np

class QLearningAgent():
    def __init__(self, s_dim, a_dim, lr=0.01, gamma=0.9, exp_noise=0.1):
        self.a_dim = a_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = exp_noise
        self.Q = np.zeros((s_dim, a_dim))

    def select_action(self,s, deterministic):
        if deterministic:
            '''deterministic policy'''
            return np.argmax(self.Q[s, :])
        else:
            '''e-greedy policy'''
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.a_dim) # e-greedy action
            else:
                return np.argmax(self.Q[s, :])

    def train(self, s, a, r, s_next, dw):
        '''Update Q table'''
        Q_sa = self.Q[s, a]
        target_Q = r + (1 - dw) * self.gamma * np.max(self.Q[s_next, :])
        self.Q[s, a] += self.lr * (target_Q - Q_sa)

    def save(self):
        '''save Q table'''
        npy_file = 'model/q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='model/q_table.npy'):
        '''load Q table'''
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')



def evaluate_policy(env, agent):
    s, info = env.reset()
    done, ep_r, steps = False, 0, 0
    while not done:
        # Take deterministic actions at test time
        a = agent.select_action(s, deterministic=True)
        s_next, r, dw, tr, info = env.step(a)
        done = (dw or tr)

        ep_r += r
        steps += 1
        s = s_next
    return ep_r