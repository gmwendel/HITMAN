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

print(f"X bounds: {np.min(P[:,0]):.3f} to {np.max(P[:,0]):.3f}")
print(f"Y bounds: {np.min(P[:,1]):.3f} to {np.max(P[:,1]):.3f}")
print(f"Z bounds: {np.min(P[:,2]):.3f} to {np.max(P[:,2]):.3f}")

print("\nMean positions (should be 0,0,0 if perfectly centered):")
print(np.mean(P, axis=0))

print("\nFirst 5 raw PMT coordinates:")
print(P[:5])
