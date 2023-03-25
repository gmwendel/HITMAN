# *HITMAN*

Likelihood reconstruction using machine learning.


## Installation


Install Tensorflow 2.10 or 2.11 using conda e.g.
```
conda create -n hitman
conda activate hitman
conda install -c anaconda python=3.7
conda install -c anaconda tensorflow-gpu=2.10.0
```

Once an anaconda environment is created, install HITMAN:
```
pip install uproot
git clone https://github.com/gmwendel/HITMAN.git
cd HITMAN
git checkout origin/comptoncamera
pip install -e .
```
Verify the scripts can be run:
```
hitman_train -h
hitman_reco -h
```



