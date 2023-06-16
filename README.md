# *HITMAN*

A tool to generate neural networks that produce likelihoods for reconstruction of physics events in optical neutrino
detectors.  This is an implementation of the technique outlined by [Eller, et. al](
https://doi.org/10.1016/j.nima.2023.168011)




## Installation

First we must set up an anaconda environment with tensorflow installed.  Attached are Linux install instructions. for 
other OS refer to [tensorflow install instructions](https://www.tensorflow.org/install/pip)
to install tensorflow on MacOS or Windows.


Create a new anaconda environment and install tensorflow 2.11 prerequisites:
```
conda deactivate
conda create -n hitman python=3.9
conda activate hitman 

conda install -c conda-forge cudatoolkit=11.2.0
pip install nvidia-cudnn-cu11==8.1.*
```

Configure the system environment to use the correct CUDA version:
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Install tensorflow
```
pip install tensorflow==2.11.*
```

Once an anaconda environment with tensorflow is properly configured, install HITMAN:
```
git clone https://github.com/gmwendel/HITMAN.git
cd HITMAN
pip install -e .
```
Verify the scripts have been added to path:
```
hitman_train -h
hitman_reco -h
```

## Basic Usage
To generate and train HITMAN networks specify 

* the path to a list of ntuple files containing training data using -i
* the location you would like to save the network using -o

e.g.
```
hitman_train -i /path/to/{1..10}_training.root.ntuple --o /path/to/netork_save_location
```

To run the likelihood reconstruction specify

* the path to a list of ntuple files containing events to be reconstructed data using -i
* the location of the network to be used for reconstruction using -n
* the location you would like to save the reconstructed events using -o

```
hitman_reco -i /path/to/{1..10}_reco.root.ntuple -n /path/to/netork_save_location -o /path/to/reco_save_location
```

Note: hitman_reco has only been verified to work for the Y90 beta spectrum and may be sensitive to likelihood spaces 
generated by different particles as hitman_reco currently uses a gradient descent algorithm to find the maximum likelihood estimation.
