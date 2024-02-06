# NoisyNet DQN-Pytorch
This is a **clean and robust Pytorch implementation of NoisyNet DQN**. A quick render here:


<img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/Render_CVP1.gif" width="90%" height="auto">  | <img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/Render%20of%20DDQN.gif" width="90%" height="auto">
:-----------------------:|:-----------------------:|

<img src="https://github.com/XinJingHao/NoisyNet-DQN-Pytorch/blob/main/IMGs/cpv1.png" width="90%" height="auto">  | <img src="https://github.com/XinJingHao/NoisyNet-DQN-Pytorch/blob/main/IMGs/lldv2.png" width="90%" height="auto">
:-----------------------:|:-----------------------:|


**Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**



## Dependencies
```python
gymnasium==0.29.1
numpy==1.26.1
pytorch==2.1.0

python==3.11.5
```

## How to use my code
### Train from scratch:
```bash
python main.py # Train NoisyNet DQN on CartPole-v1 from scratch
```

### Change Enviroment:
If you want to train on different enviroments
```bash
python main.py --EnvIdex 1
```
The --EnvIdex can be set to be 0 and 1, where   
```bash
'--EnvIdex 0' for 'CartPole-v1'  
'--EnvIdex 1' for 'LunarLander-v2'   
```
Note: if you want train on LunarLander-v2, you need to install [box2d-py](https://gymnasium.farama.org/environments/box2d/) first. You can install box2d-py via:
```bash
pip install gymnasium[box2d]
```

### Play with trained model:
```bash
python main.py --EnvIdex 0 --render True --Loadmodel True --ModelIdex 100 # Play CartPole-v1 with NoisyNet DQN
```
```bash
python main.py --EnvIdex 1 --render True --Loadmodel True --ModelIdex 550 # Play LunarLander-v2 with NoisyNet DQN
```

### Visualize the training curve
You can use the [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to record anv visualize the training curve. 

- Installation (please make sure Pytorch is installed already):
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
DQN: Mnih V , Kavukcuoglu K , Silver D , et al. Playing Atari with Deep Reinforcement Learning[J]. Computer Science, 2013. 

NoisyNet DQN: Fortunato M, Azar M G, Piot B, et al. Noisy networks for exploration[J]. arXiv preprint arXiv:1706.10295, 2017.
