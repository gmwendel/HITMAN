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
    charges, charge_hyp, pmt_positions = extractor.get_poisson_train_data()
    
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
