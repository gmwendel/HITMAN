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

unique_pmts, indices = np.unique(P, axis=0, return_index=True)
P = unique_pmts
D = D[indices]

def hash_lattice(pos, norm):
    combined = np.hstack([pos, norm])
    combined = np.round(combined, 2)
    hashed = sorted([tuple(row) for row in combined])
    return hashed

ref = hash_lattice(P, D)

matrices = [
    # E
    np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
    # C2(z)
    np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]),
    # sigma_x
    np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
    # sigma_y
    np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]),
    # S4
    np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., -1.]]),
    np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]]),
    # C2(x)
    np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]),
    # sigma_d
    np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]),
]

names = ["E", "C2(z)", "sigma_x", "sigma_y", "S4(1)", "S4(-1)", "C2(x)", "sigma_d"]

print(f"Testing manual D2d matrices...")
for name, M in zip(names, matrices):
    test = hash_lattice(P @ M.T, D @ M.T)
    if test == ref:
        print(f"SUCCESS: {name}")
    else:
        print(f"FAILED:  {name}")
        # Find the first point that doesn't match
        for i, (t, r) in enumerate(zip(test, ref)):
            if t != r:
                print(f"  First mismatch at sorted index {i}:")
                print(f"    Expected (ref): {r}")
                print(f"    Got (test)    : {t}")
                break
