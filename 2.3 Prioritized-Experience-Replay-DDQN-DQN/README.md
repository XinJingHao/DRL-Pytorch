# Prioritized Experience Replay DDQN-Pytorch

A clean and robust implementation of [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952) with DQN/DDQN. 

<img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/Render_CVP1.gif" width="90%" height="auto">  | <img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/Render%20of%20DDQN.gif" width="90%" height="auto">
:-----------------------:|:-----------------------:|

Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).


<br/>
<br/>

## How to use my code

### Train from scratch

```bash
cd LightPriorDQN_gym0.2x # or PriorDQN_gym0.2x, PriorDQN_gym0.1x

python main.py
```

where the default enviroment is CartPole-v1.  

### Play with trained model

```bash
cd LightPriorDQN_gym0.2x # or PriorDQN_gym0.2x, PriorDQN_gym0.1x

python main.py --write False --render True --Loadmodel True --ModelIdex 50
```

### Change Enviroment

If you want to train on different enviroments

```bash
cd LightPriorDQN_gym0.2x # or PriorDQN_gym0.2x, PriorDQN_gym0.1x

python main.py --EnvIdex 1
```

The --EnvIdex can be set to be 0 and 1, where   

```bash
'--EnvIdex 0' for 'CartPole-v1'  
'--EnvIdex 1' for 'LunarLander-v2'   
```

if you want train on LunarLander-v2, you need to install [box2d-py](https://gymnasium.farama.org/environments/box2d/) first.


### Visualize the training curve

You can use the [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to visualize the training curve. History training curve is saved at '\runs'

### Hyperparameter Setting

For more details of Hyperparameter Setting, please check 'main.py'

<br/>

## Versions
### This repository contains three versions of PER :
- V1: PriorDQN_gym0.1x
- V2: PriorDQN_gym0.2x
- V3: LightPriorDQN_gym0.2x

where **V3 is most recommended**, because it is the newest, simplest, and fastest one.

### Details of V1, V2, and V3:
+ **V1: PriorDQN_gym0.1x**

  Implemented with **gym==0.19.0**, where ***s_next, a, r, done, info = env.step(a)***

  Prioritized sampling is realized by ***sum-tree***

  ```python
  # Dependencies of PriorDQN_gym0.1x
  gym==0.19.0
  numpy==1.21.6
  pytorch==1.11.0
  tensorboard==2.13.0

  python==3.9.0
  ```

  |                           CartPole                           |                         LunarLander                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.1x/IMGs/CPV1.svg" width="320" height="200"> | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.1x/IMGs/LLDV2.svg" width="320" height="200"> |

<br/>
<br/>



+ **V2: PriorDQN_gym0.2x**

  Implemented with **gymnasium==0.29.1**, where ***s_next, a, r, terminated, truncated, info = env.step(a)***

  Prioritized sampling is realized by ***sum-tree***

  ```python
  # Dependencies of PriorDQN_gym0.2x
  gymnasium==0.29.1
  box2d-py==2.3.5
  numpy==1.26.1
  pytorch==2.1.0
  tensorboard==2.15.1

  python==3.11.5
  ```

  |                           CartPole                           |                         LunarLander                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.2x/IMGs/CPV1.svg" width="320" height="200"> | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.2x/IMGs/LLDV2.svg" width="320" height="200"> |

<br/>
<br/>



+ **V3: LightPriorDQN_gym0.2x**

  An optimized version of PriorDQN_gym0.2x,

  where prioritized sampling is realized by ***torch.multinomial()***, which is 3X faster than sum-tree.

  ```python
  # Dependencies of LightPriorDQN_gym0.2x
  gymnasium==0.29.1
  box2d-py==2.3.5
  numpy==1.26.1
  pytorch==2.1.0
  tensorboard==2.15.1

  python==3.11.5
  ```

  |                           CartPole                           |                         LunarLander                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/LightPriorDQN_gym0.2x/IMGs/CPV1.svg" width="320" height="200"> | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/LightPriorDQN_gym0.2x/IMGs/LLDV2.svg" width="320" height="200"> |
  
  <br/>
  
  The traning time comparasion between *LightPriorDQN_gym0.2x(red)* and *PriorDQN_gym0.2x(blue)* is given as follow, where 3X acceleration can be observed:
  
  <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/LightPriorDQN_gym0.2x/IMGs/time_comparing.svg" width="320" height="200">


<br/>
<br/>


## References

PER: Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv preprint arXiv:1511.05952, 2015.

DQN: Mnih V , Kavukcuoglu K , Silver D , et al. Playing Atari with Deep Reinforcement Learning[J]. Computer Science, 2013. 

Double DQN: Hasselt H V , Guez A , Silver D . Deep Reinforcement Learning with Double Q-learning[J]. Computer ence, 2015.

  
