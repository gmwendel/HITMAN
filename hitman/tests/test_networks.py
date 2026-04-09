import pytest
import tensorflow as tf
import numpy as np
from hitman.neural_nets.chargenet import get_chargenet
from hitman.neural_nets.hitnet import get_hitnet

def test_neural_networks_architecture():
    # Provide dummy normalization matrices
    hyp_norm = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    obs_norm_charge = np.array([[1.0, 1.0], [0.0, 0.0]])
    obs_norm_hit = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])

    # Instantiate ChargeNet
    chargenet = get_chargenet(hyp_norm=hyp_norm, obs_norm=obs_norm_charge)
    
    # 2 charge inputs + 3 parameter inputs
    assert chargenet.inputs[0].shape == (None, 2)
    assert chargenet.inputs[1].shape == (None, 3)

    # Test forward pass with dummy data
    dummy_charge = tf.zeros((1, 2))
    dummy_params = tf.zeros((1, 3))
    out_charge = chargenet([dummy_charge, dummy_params])
    assert out_charge.shape == (1, 1)

    # Instantiate HitNet
    hitnet = get_hitnet(hyp_norm=hyp_norm, obs_norm=obs_norm_hit)

    # 5 hit inputs + 3 parameter inputs
    assert hitnet.inputs[0].shape == (None, 5)
    assert hitnet.inputs[1].shape == (None, 3)

    # Test forward pass with dummy data
    dummy_hit = tf.zeros((1, 5))
    out_hit = hitnet([dummy_hit, dummy_params])
    assert out_hit.shape == (1, 1)
