<div align=center>
<img src="https://github.com/XinJingHao/RL-Algorithms-by-Pytorch/blob/main/RL_PYTORCH.png" width=500 />
</div>

<div align=center>
Clean, Robust, and Unified PyTorch implementation of popular DRL Algorithms
</div>

<div align=center>
  <img src="https://img.shields.io/badge/Python-blue" />
  <img src="https://img.shields.io/badge/Pytorch-ff69b4" />
  <img src="https://img.shields.io/badge/DRL-blueviolet" />
</div>

<br/>
<br/>

## 0.Star History

<div align="left">
<img width="70%" height="auto" src="https://api.star-history.com/svg?repos=XinJingHao/Deep-Reinforcement-Learning-Algorithms-with-Pytorch&type=Date">
</div>
<br/>


## 1.Dependencies
This repository uses the following python dependencies unless explicitly stated:
```python
gymnasium==0.29.1
numpy==1.26.1
pytorch==2.1.0

python==3.11.5
```

<br/>

## 2.How to use my code
Enter the folder of the algorithm that you want to use, and run the **main.py** to train from scratch:
```bash
python main.py
```
For more details, please check the **README.md** file in the corresponding algorithm folder.

<br/>

## 3. Separate links of the code
+ [1.Q-learning](https://github.com/XinJingHao/Q-learning)
+ [2.1Duel Double DQN](https://github.com/XinJingHao/Duel-Double-DQN-Pytorch)
+ [2.2Noisy Duel DDQN on Atari Game](https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch)
+ [2.3Prioritized Experience Replay(PER) DQN/DDQN](https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch)
+ [2.4Categorical DQN (C51)](https://github.com/XinJingHao/C51-Categorical-DQN-Pytorch)
+ [2.5NoisyNet DQN](https://github.com/XinJingHao/NoisyNet-DQN-Pytorch)
+ [3.1Proximal Policy Optimization(PPO) for Discrete Action Space](https://github.com/XinJingHao/PPO-Discrete-Pytorch)
+ [3.2Proximal Policy Optimization(PPO) for Continuous Action Space](https://github.com/XinJingHao/PPO-Continuous-Pytorch)
+ [4.1Deep Deternimistic Policy Gradient(DDPG)](https://github.com/XinJingHao/DDPG-Pytorch)
+ [4.2Twin Delayed Deep Deterministic Policy Gradient(TD3)](https://github.com/XinJingHao/TD3-Pytorch)
+ [5.1Soft Actor Critic(SAC) for Discrete Action Space](https://github.com/XinJingHao/SAC-Discrete-Pytorch)
+ [5.2Soft Actor Critic(SAC) for Continuous Action Space](https://github.com/XinJingHao/SAC-Continuous-Pytorch)
+ [6.Actor-Sharer-Learner(ASL)](https://github.com/XinJingHao/Actor-Sharer-Learner)

<br/>

## 4. Recommended Resources for DRL
### 4.1 Simulation Environments:
+ [gym](https://www.gymlibrary.dev/) and [gymnasium](https://gymnasium.farama.org/) (Lightweight & Standard Env for DRL; Easy to start; Slow):
<div align="left">
<img width="60%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Env_images/gym.gif">
</div>
<br/>

+ [Isaac Sim](https://developer.nvidia.com/isaac/sim#isaac-lab) (NVIDIA’s physics simulation environment; GPU accelerated; Superfast):
<div align="left">
<img width="60%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Env_images/IsaacGym.gif">
</div>
<br/>

+ [Sparrow](https://github.com/XinJingHao/Sparrow-V2) (Light Weight Simulator for Mobile Robot; DRL friendly):
<div align="left">
<img width="62%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V1/render.gif">
</div>

<p align="left">
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/case_v2.gif" width="10%" height="auto"  />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/case2.gif" width="10%" height="auto" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/play.gif" width="10%" height="auto" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V3/N1.gif" width="10%" height="auto" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V3/N3.gif" width="10%" height="auto" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V3/N10.gif" width="10%" height="auto" />
</p>

<br/>

+ [ROS](https://www.ros.org/) (Popular & Comprehensive physical simulator for robots; Heavy and Slow):
<div align="left">
<img width="60%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Env_images/ros.mp4.gif">
</div>
<br/>

+ [Webots](https://cyberbotics.com/) (Popular physical simulator for robots; Faster than ROS; Less realistic):
<div align="left">
<img width="60%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Env_images/webots.gif">
</div>
<br/>

+ [Envpool](https://envpool.readthedocs.io/en/latest/index.html) (Fast Vectorized Env)
+ [Other Popular Envs](https://github.com/clvrai/awesome-rl-envs)

### 4.2 Books：
+ [《Reinforcement learning: An introduction》](https://books.google.com.sg/books?hl=zh-CN&lr=&id=uWV0DwAAQBAJ&oi=fnd&pg=PR7&dq=Reinforcement+Learning&ots=mivIu01Xp6&sig=zQ6jkZRxJop4fkAgScMgzULGlbY&redir_esc=y#v=onepage&q&f=false)--Richard S. Sutton
+ 《深度学习入门：基于Python的理论与实现》--斋藤康毅

### 4.3 Online Courses:
+ [RL Courses(bilibili)](https://www.bilibili.com/video/BV1UE411G78S?p=1&vd_source=df4b7370976f5ca5034cc18488eec368)--李宏毅(Hongyi Li)
+ [RL Courses(Youtube)](https://www.youtube.com/watch?v=z95ZYgPgXOY&list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)--李宏毅(Hongyi Li)
+ [UCL Course on RL](https://www.davidsilver.uk/teaching/)--David Silver
+ [动手强化学习](https://hrl.boyuai.com/chapter/1/%E5%88%9D%E6%8E%A2%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)--上海交通大学
+ [DRL Courses](https://github.com/wangshusen/DRL)--Shusen Wang

### 4.4 Blogs:
+ [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
+ [Policy Gradient Theorem --Cangxi](https://zhuanlan.zhihu.com/p/491647161)
+ [Policy Gradient Algorithms --Lilian](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
+ [Theorem of PPO](https://zhuanlan.zhihu.com/p/563166533)
+ [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
+ [Prioritized Experience Replay](https://zhuanlan.zhihu.com/p/631171588)
+ [Soft Actor Critic](https://zhuanlan.zhihu.com/p/566722896)
+ [A (Long) Peek into Reinforcement Learning --Lilian](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
+ [Introduction to TD3](https://zhuanlan.zhihu.com/p/409536699)

<br/>

## 5. Important Papers
DQN: [Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. nature, 2015, 518(7540): 529-533.](https://www.nature.com/articles/nature14236/?source=post_page)

Double DQN: [Van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double q-learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2016, 30(1).](https://ojs.aaai.org/index.php/AAAI/article/view/10295)

Duel DQN: [Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.](https://proceedings.mlr.press/v48/wangf16.pdf)

PER: [Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/abs/1511.05952)

C51: [Bellemare M G, Dabney W, Munos R. A distributional perspective on reinforcement learning[C]//International conference on machine learning. PMLR, 2017: 449-458.](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf)

NoisyNet DQN: [Fortunato M, Azar M G, Piot B, et al. Noisy networks for exploration[J]. arXiv preprint arXiv:1706.10295, 2017.](https://arxiv.org/abs/1706.10295)

PPO: [Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.](https://arxiv.org/pdf/1707.06347.pdf)

DDPG: [Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[J]. arXiv preprint arXiv:1509.02971, 2015.](https://arxiv.org/abs/1509.02971)

TD3: [Fujimoto S, Hoof H, Meger D. Addressing function approximation error in actor-critic methods[C]//International conference on machine learning. PMLR, 2018: 1587-1596.](https://proceedings.mlr.press/v80/fujimoto18a.html)

SAC: [Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C]//International conference on machine learning. PMLR, 2018: 1861-1870.](https://proceedings.mlr.press/v80/haarnoja18b)

ASL: [Train a Real-world Local Path Planner in One Hour via Partially Decoupled Reinforcement Learning and Vectorized Diversity](https://arxiv.org/abs/2305.04180)

ColorDynamic: [Generalizable, Scalable, Real-time, End-to-end Local Planner for Unstructured and Dynamic Environments](https://arxiv.org/abs/2502.19892)

<br/>


## 6. Training Curves of my Code:

### [Q-learning:](https://github.com/XinJingHao/Q-learning)
<img src="https://github.com/XinJingHao/Q-learning/blob/main/result.svg" width=320>

### [Duel Double DQN:](https://github.com/XinJingHao/Duel-Double-DQN-Pytorch)
|                           CartPole                           |                         LunarLander                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
<img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/cp_all.png" width="320" height="200"> | <img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/lld_all.png" width="320" height="200">



### [Noisy Duel DDQN on Atari Game:](https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch)
Pong| Enduro
:-----------------------:|:-----------------------:|
<img src="https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch/blob/main/IMGs/Pong.png" width="320" height="200">| <img src="https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch/blob/main/IMGs/Enduro.png" width="320" height="200">

<br/>


### [Prioritized DQN/DDQN:](https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch)
|                           CartPole                           |                         LunarLander                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/LightPriorDQN_gym0.2x/IMGs/CPV1.svg" width="320" height="200"> | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/LightPriorDQN_gym0.2x/IMGs/LLDV2.svg" width="320" height="200"> |

<br/>

### [Categorical DQN:](https://github.com/XinJingHao/C51-Categorical-DQN-Pytorch)
|                           CartPole                           |                         LunarLander                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/XinJingHao/C51-Categorical-DQN-Pytorch/blob/main/Images/cp.svg" width="320" height="200"> | <img src="https://github.com/XinJingHao/C51-Categorical-DQN-Pytorch/blob/main/Images/lld.svg" width="320" height="200"> |

<br/>

### [NoisyNet DQN:](https://github.com/XinJingHao/C51-Categorical-DQN-Pytorch)
|                           CartPole                           |                         LunarLander                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/XinJingHao/NoisyNet-DQN-Pytorch/blob/main/IMGs/cpv1.png" width="320" height="200"> | <img src="https://github.com/XinJingHao/NoisyNet-DQN-Pytorch/blob/main/IMGs/lldv2.png" width="320" height="200"> |

<br/>

### [PPO Discrete:](https://github.com/XinJingHao/PPO-Discrete-Pytorch)
<img src="https://github.com/XinJingHao/PPO-Discrete-Pytorch/blob/main/result.jpg" width=700>

### [PPO Continuous:](https://github.com/XinJingHao/PPO-Continuous-Pytorch)
<img src="https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/ppo_result.jpg">

### [DDPG:](https://github.com/XinJingHao/DDPG-Pytorch)
Pendulum| LunarLanderContinuous
:-----------------------:|:-----------------------:|
<img src="https://github.com/XinJingHao/DDPG-Pytorch/blob/main/IMGs/ddpg_pv0.svg" width="320" height="200">| <img src="https://github.com/XinJingHao/DDPG-Pytorch/blob/main/IMGs/ddpg_lld.svg" width="320" height="200"> 

<br/>

### [TD3:](https://github.com/XinJingHao/TD3-Pytorch)
<img src="https://github.com/XinJingHao/TD3-Pytorch/blob/main/images/TD3results.png" width=700>

### [SAC Continuous:](https://github.com/XinJingHao/SAC-Continuous-Pytorch)
<img src="https://github.com/XinJingHao/SAC-Continuous-Pytorch/blob/main/imgs/result.jpg" width=700>

### [SAC Discrete:](https://github.com/XinJingHao/SAC-Discrete-Pytorch)
<img src="https://github.com/XinJingHao/SAC-Discrete-Pytorch/blob/main/imgs/sacd_result.jpg" width=700>

### [Actor-Sharer-Learner (ASL):](https://github.com/XinJingHao/Actor-Sharer-Learner)
<div align="left">
<img width="70%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/asl/ss_e.svg">
</div>


