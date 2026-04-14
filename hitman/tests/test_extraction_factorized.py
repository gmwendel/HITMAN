import numpy as np
import pytest
from hitman.tools.ratextract_factorized import FactorizedDataExtractor
import uproot
import os

class MockDataExtractor(FactorizedDataExtractor):
    def __init__(self):
        # Bypass the file loading and setup mocked data
        self.input_files = []
        pass
        
    def mock_data(self, npes, pmt_ids, num_sensors):
        # This will be injected dynamically in the test
        self.mocked_npes = npes
        self.mocked_pmt_ids = pmt_ids
        self.mocked_num_sensors = num_sensors
        
    def _load_raw_data(self):
        return {
            'mcPMTNPE': self.mocked_npes,
            'mcPMTID': self.mocked_pmt_ids
        }
        
    def get_factorized_targets(self):
        raw_data = self._load_raw_data()
        
        flat_pmt_ids = np.concatenate(raw_data['mcPMTID']).astype(np.int32)
        flat_npes = np.concatenate(raw_data['mcPMTNPE'])
        
        N_events = len(raw_data['mcPMTID'])
        N_sensors = self.mocked_num_sensors
        
        event_lengths = np.array([len(x) for x in raw_data['mcPMTID']], dtype=np.int32)
        event_indices = np.repeat(np.arange(N_events), event_lengths)
        
        charges = np.zeros((N_events, N_sensors), dtype=np.float32)
        np.add.at(charges, (event_indices, flat_pmt_ids), flat_npes)
        
        injected_yields = np.ones(N_events, dtype=np.float32)
        
        return self._process_charges(charges, injected_yields)

def test_factorized_extractor_normalization():
    extractor = MockDataExtractor()
    
    # Event 0: Some hits
    # Event 1: No hits at all
    npes = np.array([[10, 20], []], dtype=object)
    pmt_ids = np.array([[0, 1], []], dtype=object)
    
    extractor.mock_data(npes, pmt_ids, 3) # 3 total sensors
    
    shape_target, rate_target = extractor.get_factorized_targets()
    
    # Check rate targets
    assert np.allclose(rate_target, np.array([30, 0]))
    
    # Check shape targets
    # Summing over the sensors for each event should equal 1.0 (PMF constraint)
    assert np.allclose(np.sum(shape_target, axis=1), np.array([1.0, 1.0]))
    
    # Check that the empty sensor got the Laplace smoothing
    assert shape_target[0, 2] > 0.0
    
    # Ensure there are no NaNs
    assert not np.isnan(shape_target).any()
