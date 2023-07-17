# 3-D SPiEL 
<h3 align="center">3-Dimensional Sonic Phase-invariant Echo Localization</h3>
<p align="center"><a href="http://www.christopherhahne.de/"><strong>Christopher Hahne</strong></a></p>
<p align="center">University of Bern, Switzerland</p>
<p align="center"><strong>ICRA'23, London, UK</strong></p>
<br>


[![preview](https://user-images.githubusercontent.com/33809838/233854687-1fbf6ac4-91e6-4640-908d-a66c4beff66e.png)](https://arxiv.org/abs/2306.08281)


## Installation
Make sure your ```python3 --version``` is at least ```3.8.x```. Then run
```
$ bash install_env.sh
```

If the installation fails, please revise and adapt [the above script](https://github.com/hahnec/spiel/blob/master/install_env.sh) according to your preferred environment.

## Run
First, activate your environment, e.g.
```
$ source venv/bin/activate
```

For results computation, use
```
$ python eval.py
```

Training of the MLP is done with
```
$ python train_spiel.py
```

## Paper

If you use this project for your work, please cite the original paper:

[![arXiv paper link](https://img.shields.io/badge/paper-arXiv:2306.08281-red)](https://arxiv.org/pdf/2306.08281.pdf)
```
@inproceedings{spiel:icra23,
  title={3-Dimensional Sonic Phase-invariant Echo Localization},
  author={Hahne, Christopher},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
  pages={4121-4127},
  doi={10.1109/ICRA48891.2023.10161199}
}
```

## Acknowledgment
This study was funded by the Hasler Foundation under project number 22027.
