# DQN/DDQN-Atari-Pytorch
This is a **clean and robust Pytorch implementation of DQN and Double DQN and their applications on Atari**.

Pong| Enduro
:-----------------------:|:-----------------------:|
<img src="https://github.com/XinJingHao/DQN-DDQN-Atari-Pytorch/raw/main/IMGs/Pong.gif" width="320" height="480">| <img src="https://github.com/XinJingHao/DQN-DDQN-Atari-Pytorch/raw/main/IMGs/Enduro.gif" width="320" height="480">
<img src="https://github.com/XinJingHao/DQN-DDQN-Atari-Pytorch/raw/main/IMGs/Pong.png" width="320" height="200">| <img src="https://github.com/XinJingHao/DQN-DDQN-Atari-Pytorch/raw/main/IMGs/Enduro.png" width="320" height="200">

All the experiments are trained with same hyperparameters. **Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**



## Dependencies
```bash
numpy == 1.21.6  
pytorch == 1.11.0  
gym == 0.19.0  
atari_py == 0.2.6  
```

P.S. A instruction on how to install Atari can be found [here](https://zhuanlan.zhihu.com/p/523895071)

## How to use my code
### Train from scratch
```bash
python main.py
```
where the default enviroment is PongNoFrameskip-v4. 

### Play with trained model
```bash
python main.py --render True --Loadmodel True --ModelIdex 900
```
### Change Enviroment
If you want to train on different enviroments, just run 
```bash
python main.py --EnvIdex 0
```
The ```--EnvIdex``` can be set to be 0 and 1, where 
```bash
'--EnvIdex 0' for 'EnduroNoFrameskip-v4'  
'--EnvIdex 1' for 'PongNoFrameskip-v4' 
```

### Visualize the training curve
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

### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'

### References
DQN: Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. nature, 2015, 518(7540): 529-533.

Double DQN: Hasselt H V , Guez A , Silver D . Deep Reinforcement Learning with Double Q-learning[J]. Computer ence, 2015.

