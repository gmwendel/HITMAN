import pytest
import numpy as np
import os
from hitman.tools.ratextract_poisson import PoissonDataExtractor

def test_poisson_data_extraction():
    test_file = os.path.abspath("../../data/sbi_test_reco_320/job_00000.root")
    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found. Skipping extraction test.")

    extractor = PoissonDataExtractor([test_file])
    
    # Test train data extraction
    charges, charge_hyp, pmt_positions, hit_obs, hit_hyp = extractor.get_poisson_train_data()
    
    N_events = len(charge_hyp)
    N_sensors = len(pmt_positions)
    
    assert charges.shape == (N_events, N_sensors), f"Expected charges shape {(N_events, N_sensors)}, got {charges.shape}"
    assert charge_hyp.shape == (N_events, 3), f"Expected charge_hyp shape {(N_events, 3)}, got {charge_hyp.shape}"
    assert pmt_positions.shape == (N_sensors, 3), f"Expected pmt_positions shape {(N_sensors, 3)}, got {pmt_positions.shape}"
    
    # Test reco data extraction
    events = extractor.get_poisson_reco_data()
    assert len(events) == N_events
    
    event = events[0]
    assert event['charges'].shape == (N_sensors,)
    assert event['truth'].shape == (3,)
    assert event['pmt_positions'].shape == (N_sensors, 3)

def test_indom_shuffling():
    from hitman.tools.datagenerator import DataGenerator
    # Dummy data: 3 PMT hits (x,y,z,t,charge) and corresponding true params
    # 2 hits on PMT A (1,1,1), 1 hit on PMT B (2,2,2)
    x = np.array([
        [1.0, 1.0, 1.0, 10.0, 1.0], # PMT A, event 1
        [1.0, 1.0, 1.0, 20.0, 1.0], # PMT A, event 2
        [2.0, 2.0, 2.0, 30.0, 1.0]  # PMT B, event 3
    ])
    
    t = np.array([
        [0.1, 0.2, 3000.0], # hyp 1
        [0.4, 0.5, 6000.0], # hyp 2
        [0.9, 0.8, 9000.0]  # hyp 3
    ])
    
    # Run inDOM shuffling
    gen = DataGenerator(x, t, batch_size=2, shuffle='inDOM', time_spread=0)
    shuffled_t = gen.shuffled_params
    
    # PMT B only has 1 hit, so its hypothesis MUST remain identical (cannot swap with PMT A)
    np.testing.assert_array_equal(shuffled_t[2], t[2])
    
    # PMT A has 2 hits, its hypotheses should be swapped or identical, but restricted to hyp 1 and hyp 2
    # Check that rows 0 and 1 of shuffled_t are some permutation of rows 0 and 1 of t
    valid_hyp_A = [t[0].tolist(), t[1].tolist()]
    assert shuffled_t[0].tolist() in valid_hyp_A
    assert shuffled_t[1].tolist() in valid_hyp_A
