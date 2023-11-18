## Actor-Sharer-Learner (ASL): An Efficient Training Framework for Off-policy Deep Reinforcement Learning
<div align="center">
  <a ><img width="1200px" height="auto" src="https://github.com/XinJingHao/Images/blob/main/asl/ASL.jpg"></a>
</div>

![Python](https://img.shields.io/badge/Python-blue)
![Pytorch](https://img.shields.io/badge/Pytorch-ff69b4)
![DRL](https://img.shields.io/badge/DRL-blueviolet)
![TrainingFramework](https://img.shields.io/badge/TrainingFramework-9cf)

## Introduction
The [Actor-Sharer-Learner (ASL)](https://arxiv.org/abs/2305.04180) is a highly efficient training framework for off-policy DRL algorithms, capable of enhancing sample efficiency, shortening training time, and improving final performance simultaneously. Detailly, the ASL framework employs a Vectorized Data Collection (VDC) mode to expedite data acquisition, decouples the data collection from model optimization by multithreading, and partially connects the two procedures by harnessing a Time Feedback Mechanism (TFM) to evade data underuse or overuse.

## Dependencies
```bash
envpool >= 0.6.6  (https://envpool.readthedocs.io/en/latest/)
torch >= 1.13.0  (https://pytorch.org/)
numpy >= 1.23.4  (https://numpy.org/)
tensorboard >= 2.11.0  (https://pytorch.org/docs/stable/tensorboard.html)
python >= 3.8.0 
ubuntu >= 18.04.1 
```

## Quick Start:
After installation, you can use the ASL framework to train an Atari agent via:
```bash
python main.py
```
where the default envionment is **Alien** and the underlying DRL algorithm is **DDQN**. For more details about experiment setup, please check the **main.py**. The trianing curves of 57 [Atari](https://www.gymlibrary.dev/environments/atari/) games are listed as follows.

<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/asl/ss_e.svg">
</div>


## Citing the Project

To cite this repository in publications:

```bibtex
@article{Color2023JinghaoXin,
  title={Train a Real-world Local Path Planner in One Hour via Partially Decoupled Reinforcement Learning and Vectorized Diversity},
  author={Jinghao Xin, Jinwoo Kim, Zhi Li, and Ning Li},
  journal={arXiv preprint arXiv:2305.04180},
  url={https://doi.org/10.48550/arXiv.2305.04180},
  year={2023}
}
```

## Maintenance History
+ 2023/6/20
  + `sample_core()` in `Sharer.py` is optimized, where
    + we use a more pytorch way to delete `self.ptr-1` in `ind`
    + for `Sharer.shared_data_cuda()`, the `ind` and `env_ind` are generated on `self.B_dvc` to run faster


