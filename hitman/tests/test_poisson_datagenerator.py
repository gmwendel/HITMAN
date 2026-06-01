import numpy as np
import tensorflow as tf
import pytest

def test_poisson_dataset_shapes():
    try:
        from hitman.tools.poisson_datagenerator import get_poisson_dataset
    except ImportError:
        pytest.fail("get_poisson_dataset not implemented yet")

    N_events = 100
    N_sensors = 128
    batch_size = 256
    
    charges = np.random.poisson(1.5, size=(N_events, N_sensors)).astype(np.float32)
    charge_hyp = np.random.uniform(size=(N_events, 3)).astype(np.float32)
    pmt_positions = np.random.uniform(size=(N_sensors, 3)).astype(np.float32)
    
    dataset = get_poisson_dataset(charges, charge_hyp, pmt_positions, batch_size=batch_size)
    
    assert isinstance(dataset, tf.data.Dataset), "Should return a tf.data.Dataset"
    
    for (batch_pmt, batch_hyp), batch_charges in dataset.take(1):
        assert batch_pmt.shape == (batch_size, 3)
        assert batch_hyp.shape == (batch_size, 3)
        assert batch_charges.shape == (batch_size,)
        break

def test_validation_split():
    # Test that we can use take and skip on the base event dataset
    from hitman.tools.poisson_datagenerator import get_poisson_dataset
    
    N_events = 100
    N_sensors = 128
    batch_size = 256
    
    charges = np.arange(N_events * N_sensors).reshape((N_events, N_sensors)).astype(np.float32)
    charge_hyp = np.arange(N_events * 3).reshape((N_events, 3)).astype(np.float32)
    pmt_positions = np.random.uniform(size=(N_sensors, 3)).astype(np.float32)

    # We must be able to instruct get_poisson_dataset to build a pipeline from an existing Dataset
    # OR we handle the splits via arguments. Let's assume we update the function to accept a base dataset or split parameters.
    # We will test the functionality that take/skip is applied before shuffle to prevent leakage.
    
    # We expect get_poisson_dataset to accept `take` or `skip` arguments, or `validation_split` fraction.
    # Let's mock the expected API: `get_poisson_dataset(..., split='train', val_fraction=0.1)`
    
    train_ds = get_poisson_dataset(charges, charge_hyp, pmt_positions, batch_size=batch_size, shuffle=False, split='train', val_fraction=0.1)
    val_ds = get_poisson_dataset(charges, charge_hyp, pmt_positions, batch_size=batch_size, shuffle=False, split='val', val_fraction=0.1)
    
    train_charges = []
    for _, batch_charges in train_ds:
        train_charges.extend(batch_charges.numpy())
        
    val_charges = []
    for _, batch_charges in val_ds:
        val_charges.extend(batch_charges.numpy())
        
    # Check that there is no overlap
    train_set = set(train_charges)
    val_set = set(val_charges)
    assert len(train_set.intersection(val_set)) == 0, "Data leakage between train and val splits!"
    assert len(val_charges) == 10 * N_sensors # 10% of 100 events = 10 events
    assert len(train_charges) == 90 * N_sensors # 90% of 100 events = 90 events

