# Noisy-Duel-DDQN-Atari-Pytorch
This is a **clean and robust Pytorch implementation of Noisy-Duel-DDQN on Atari**.

Pong| Enduro
:-----------------------:|:-----------------------:|
<img src="https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch/blob/main/IMGs/Pong.gif" width="320" height="480">| <img src="https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch/blob/main/IMGs/Enduro.gif" width="320" height="480">
<img src="https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch/blob/main/IMGs/Pong.png" width="320" height="200">| <img src="https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch/blob/main/IMGs/Enduro.png" width="320" height="200">

All the experiments are trained with same hyperparameters. **Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**



## Dependencies
```bash
gymnasium==0.29.1
numpy==1.26.1
pytorch==2.1.0

python==3.11.5
```

P.S. You can install the Atari environment via ```pip install gymnasium[atari] gymnasium[accept-rom-license]```

## How to use my code
### Train from scratch:
```bash
python main.py # Train PongNoFrameskip-v4 with DQN
```

### Change Algorithm:
```bash
python main.py --Double True --Duel False --Noisy False # Use Double DQN
```
```bash
python main.py --Double False --Duel True --Noisy False # Use Duel DQN
```
```bash
python main.py --Double False --Duel False --Noisy True # Use Noisy DQN
```
```bash
python main.py --Double True --Duel True --Noisy True # Use Double Duel Noisy DQN
```

### Change Enviroment:
If you want to train on different enviroments, just run 
```bash
python main.py --EnvIdex 20 # Train EnduroNoFrameskip-v4 with DQN
```
The ```--EnvIdex``` can be set to be 1~57, where 
```python
1: "Alien",
2: "Amidar",
...
20: "Enduro",
...
57: "Zaxxon"
```
For more details, please refer to ```AtariNames.py```. 

Note that the hyperparameters of this code is a light version (we only use a replay buffer of size 10000 to save memory). Thus, the default hyperparameters may not perform well on all the games. If you want a more robost hyperparameters, please check the [DQN paper (Nature)](https://www.nature.com/articles/nature14236/?source=post_page).


### Play with trained model:
```bash
python main.py --render True --EnvIdex 20 --Double True --Duel True --Noisy False --Loadmodel True --ModelIdex 900 # Play with Enduro
```
```bash
python main.py --render True --EnvIdex 37 --Double True --Duel True --Noisy True --Loadmodel True --ModelIdex 700 # Play with Pong
```

### Visualize the training curve:
You can use the [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to record anv visualize the training curve. 

- Installation (please make sure PyTorch is installed already):
```bash
pip install tensorboard
pip install packaging
```
- Record (the training curves will be saved at '**\runs**'):
```bash
python main.py --write True
```

- Visualization:
```bash
tensorboard --logdir runs
```

### Hyperparameter Setting:
For more details of Hyperparameter Setting, please check 'main.py'

### References:
DQN: [Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. nature, 2015, 518(7540): 529-533.](https://www.nature.com/articles/nature14236/?source=post_page)

Double DQN: [Van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double q-learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2016, 30(1).](https://ojs.aaai.org/index.php/AAAI/article/view/10295)

Duel DQN: [Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.](https://proceedings.mlr.press/v48/wangf16.pdf)

NoisyNet DQN: [Fortunato M, Azar M G, Piot B, et al. Noisy networks for exploration[J]. arXiv preprint arXiv:1706.10295, 2017.](https://arxiv.org/abs/1706.10295)

