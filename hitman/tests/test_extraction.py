import pytest
import numpy as np
import os
from hitman.tools.ratextract import DataExtractor

def test_data_extraction_and_normalization():
    # Use one of the generated ROOT files for testing
    test_file = os.path.abspath("../../data/sbi_dataset_large/job_00000.root")
    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found. Skipping extraction test.")

    extractor = DataExtractor([test_file])
    charge_obs, hit_obs, charge_hyp, hit_hyp = extractor.get_hitman_train_data()

    # Validate output shapes and lengths
    assert len(charge_obs) == len(charge_hyp)
    assert charge_hyp.shape[1] == 3, f"Expected charge_hyp to have 3 features, got {charge_hyp.shape[1]}"
    assert hit_hyp.shape[1] == 3, f"Expected hit_hyp to have 3 features, got {hit_hyp.shape[1]}"
    
    # Calculate norms dynamically as done in train_hitman.py
    charge_hyp_norm = np.stack([np.std(charge_hyp, axis=0), np.mean(charge_hyp, axis=0)])
    charge_hyp_norm[0][charge_hyp_norm[0] == 0] = 1.0

    # Ensure the norms are correctly calculated
    assert charge_hyp_norm.shape == (2, 3), f"Expected norm shape (2, 3), got {charge_hyp_norm.shape}"

    # Normalize a sample
    sample_normed = (charge_hyp - charge_hyp_norm[1]) / charge_hyp_norm[0]
    
    # Ensure mean is roughly 0 and std is roughly 1 after normalization
    assert np.allclose(np.mean(sample_normed, axis=0), 0, atol=1e-5), "Normalized mean should be 0"
    assert np.allclose(np.std(sample_normed, axis=0), 1, atol=1e-5), "Normalized std should be 1"
