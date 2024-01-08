from matplotlib import pyplot as plt
import torch

def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)

def render_policy(env, agent, opt):
    plt.ion() # Dynamic Plot
    x_range = torch.linspace(opt.v_min, opt.v_max, steps=opt.n_atoms).numpy()
    width = (opt.v_max - opt.v_min)/(opt.n_atoms-1)
    while True:
        s, info = env.reset()
        done, Episodic_scores = False, 0
        with torch.no_grad():
            while not done:
                s = torch.FloatTensor(s.reshape(1, -1)).to(opt.dvc)
                distributions, q_values = agent.q_net._predict(s)
                a = torch.argmax(q_values, dim=1).cpu().item()
                s_next, r, dw, tr, info = env.step(a)
                done = (dw or tr)
                Episodic_scores += r
                s = s_next

                dists = distributions.squeeze().cpu().numpy() # (a_dim, n_atoms)
                for i in range(opt.action_dim):
                    plt.bar(x_range, dists[i], width=width, label=opt.action_info[opt.EnvIdex][i])
                plt.ylim(0, 0.6)
                plt.legend(loc='upper left')
                plt.title('C51 by XinJingHao')
                plt.pause(0.00001)  # 暂停一段时间，不然画的太快会卡住显示不出来
                plt.clf()  # 清除之前画的图

        print('Episodic scores:', int(Episodic_scores))



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
        print('Wrong Input.')
        raise