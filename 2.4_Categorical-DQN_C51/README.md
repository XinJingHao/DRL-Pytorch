# C51: Categorical-DQN-Pytorch
A **clean and robust Pytorch implementation of Categorical DQN(C51)** ：

Render | Training curve
:-----------------------:|:-----------------------:|
<img src="https://github.com/XinJingHao/C51-Categorical-DQN-Pytorch/blob/main/Images/lld.gif" width="80%" height="auto">  | <img src="https://github.com/XinJingHao/C51-Categorical-DQN-Pytorch/blob/main/Images/lld.svg" width="100%" height="auto">

**Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**



## Dependencies
```python
gymnasium==0.29.1
matplotlib==3.8.2
numpy==1.26.1
pytorch==2.1.0

python==3.11.5
```

## How to use my code
### Train from scratch
```bash
python main.py
```
where the default enviroment is 'CartPole'.  

### Change Enviroment
If you want to train on different enviroments, just run:
```bash
python main.py --EnvIdex 1
```

The --EnvIdex can be set to be 0 and 1, where   
```bash
'--EnvIdex 0' for 'CartPole-v1'  
'--EnvIdex 1' for 'LunarLander-v2'
```

Note: if you want to play on LunarLander, you need to install [box2d-py](https://gymnasium.farama.org/environments/box2d/) first. You can install box2d-py via: ```pip install gymnasium[box2d]```


### Play with trained model
```bash
python main.py --EnvIdex 0 --render True --Loadmodel True --ModelIdex 60 # Play with CartPole
```
```bash
python main.py --EnvIdex 1 --render True --Loadmodel True --ModelIdex 320 # Play with LunarLander
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
[Bellemare M G, Dabney W, Munos R. A distributional perspective on reinforcement learning[C]//International conference on machine learning. PMLR, 2017: 449-458.](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf)
