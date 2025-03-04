import torch
import numpy as np

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

def quantile_huber_loss(quantiles, target_quantiles, kappa):
    diff = target_quantiles.unsqueeze(0) - quantiles.unsqueeze(1)
    loss = torch.where(
        diff < 0,
        (1 - kappa) * diff ** 2,
        kappa * diff ** 2
    )
    return loss.mean()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
