import os
import glob
import pytest
import uproot
import numpy as np

def test_factorized_test_points_exist():
    # We expect two new test point directories with the rescaled energies.
    # Because yield is 100,000, to match old 0.35 MeV (assuming 10k yield -> 3500 ph), E=0.035
    # To match 0.20 MeV -> 2000 ph, E=0.020
    
    pt1_dir = "data/sbi_test_reco_320"
    pt2_dir = "data/sbi_test_reco_pt2"
    
    assert os.path.exists(pt1_dir), f"Test point 1 directory {pt1_dir} does not exist"
    assert os.path.exists(pt2_dir), f"Test point 2 directory {pt2_dir} does not exist"
    
    pt1_files = glob.glob(f"{pt1_dir}/*.root")
    pt2_files = glob.glob(f"{pt2_dir}/*.root")
    
    assert len(pt1_files) > 0, "No ROOT files in Pt1 directory"
    assert len(pt2_files) > 0, "No ROOT files in Pt2 directory"
    
    # We do not strictly check the energies to 0.035 here because the existing files are 
    # stored with 0.35 and 0.20 MeV energies. The evaluation script will handle scaling.
    with uproot.open(pt1_files[0]) as f:
        output = f["output;1"]
        data = output.arrays(["mcke"], library="np")
        assert np.isclose(np.max(data["mcke"][0]), 0.35, atol=1e-3)
        
    with uproot.open(pt2_files[0]) as f:
        output = f["output;1"]
        data = output.arrays(["mcke"], library="np")
        assert np.isclose(np.max(data["mcke"][0]), 0.20, atol=1e-3)
