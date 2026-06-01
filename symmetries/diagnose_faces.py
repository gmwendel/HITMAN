import uproot
import glob
import numpy as np

files = glob.glob("/tank/BPULSE/Simulations/data/realistic_test_pt2/*.root")
f = uproot.open(files[0])
meta = f["meta"]
P = np.stack([
    meta["pmtX"].array(library="np")[0],
    meta["pmtY"].array(library="np")[0],
    meta["pmtZ"].array(library="np")[0]
], axis=1)

D = np.stack([
    meta["pmtU"].array(library="np")[0],
    meta["pmtV"].array(library="np")[0],
    meta["pmtW"].array(library="np")[0]
], axis=1)

x_faces = np.abs(P[:, 0]) > 90
y_faces = np.abs(P[:, 1]) > 90
z_faces = np.abs(P[:, 2]) > 69

print(f"PMTs on X faces: {np.sum(x_faces)}")
print(f"PMTs on Y faces: {np.sum(y_faces)}")
print(f"PMTs on Z faces: {np.sum(z_faces)}")
