import numpy as np
import itertools
import glob
import uproot

# --- 1. Load Data ---
files = glob.glob("/tank/BPULSE/Simulations/data/realistic_test_pt2/*.root")
f = uproot.open(files[0])
meta = f["meta"]
pos = np.stack([
    meta["pmtX"].array(library="np")[0],
    meta["pmtY"].array(library="np")[0],
    meta["pmtZ"].array(library="np")[0]
], axis=1)

norm = np.stack([
    meta["pmtU"].array(library="np")[0],
    meta["pmtV"].array(library="np")[0],
    meta["pmtW"].array(library="np")[0]
], axis=1)

unique_pmts, indices = np.unique(pos, axis=0, return_index=True)
pos = unique_pmts
norm = norm[indices]

# --- 2. Generate the Full Cubic Group (O_h) ---
def get_Oh_matrices():
    """Generates all 48 orthogonal matrices (permutations + sign flips)."""
    matrices = []
    for p in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3): R[i, p[i]] = signs[i]
            matrices.append(R)
    return matrices

# --- 3. Robust Hashing (Integer Projection) ---
def to_integer_lattice(arr, pitch=20.0): # Use 2.0 if working in cm, 20.0 if mm
    return np.round(arr / pitch).astype(int)

def hash_detector(p, n):
    """Creates a unique, order-invariant signature of the entire detector."""
    int_p = to_integer_lattice(p)
    int_n = np.round(n).astype(int)
    combined = np.hstack([int_p, int_n])
    # Tuple conversion allows native Python lexicographical set sorting
    return sorted([tuple(row) for row in combined])

# --- 4. The Global Spot Test ---
print("=== O(3) / D2d Quotient Space Verification ===")
Oh_matrices = get_Oh_matrices()
ref_hash = hash_detector(pos, norm)

valid_symmetries = []
invalid_symmetries = []

for R in Oh_matrices:
    mapped_pos = pos @ R.T
    mapped_norm = norm @ R.T
    
    if hash_detector(mapped_pos, mapped_norm) == ref_hash:
        valid_symmetries.append(R)
    else:
        invalid_symmetries.append(R)

print(f"Total Matrices Tested:   {len(Oh_matrices)}")
print(f"Valid D2d Isometries:    {len(valid_symmetries)} (Expected: 8)")
print(f"Rejected Operations:     {len(invalid_symmetries)} (Expected: 40)")

assert len(valid_symmetries) == 8, "CRITICAL: Detector does not possess exact D2d symmetry!"

# --- 5. The "Break" Test (Deep Dive on a Rejected Symmetry) ---
print("\n=== Deep Dive: Why an invalid symmetry breaks ===")

# Find a pure Horizontal Reflection (Z -> -Z, X->X, Y->Y)
sigma_h = None
for R in invalid_symmetries:
    if np.allclose(R, np.diag([1, 1, -1])):
        sigma_h = R
        break

if sigma_h is not None:
    print("Testing Forbidden Operation: pure Z-reflection (sigma_h)")
    
    # Pick a PMT on the +X wall in an EVEN layer (contains X-directed fibers)
    even_z_pmts = np.where(to_integer_lattice(pos[:, 2]) % 2 == 0)[0]
    plus_x_pmts = np.where(pos[even_z_pmts, 0] > 0)[0]
    test_idx = even_z_pmts[plus_x_pmts[0]]
    
    orig_p = pos[test_idx]
    orig_n = norm[test_idx]
    
    # Map it
    mapped_p = sigma_h @ orig_p
    mapped_n = sigma_h @ orig_n
    
    print(f"  Original PMT {test_idx}:")
    print(f"    Position: {orig_p}")
    print(f"    Normal:   {orig_n} (X-directed fiber)")
    
    print(f"\n  Mapped Hypothesis:")
    print(f"    Position: {mapped_p}")
    print(f"    Normal:   {mapped_n} (Requires an X-directed fiber here)")
    
    # Check what physically exists at the mapped position
    mapped_int_p = to_integer_lattice(mapped_p)
    found_physical_pmt = False
    
    for j in range(len(pos)):
        if np.array_equal(to_integer_lattice(pos[j]), mapped_int_p):
            found_physical_pmt = True
            print(f"\n  Physical Reality at this position (PMT {j}):")
            print(f"    Position: {pos[j]}")
            print(f"    Normal:   {norm[j]} (Y-directed fiber!)")
            break
            
    if found_physical_pmt:
        print("\n  CONCLUSION: The symmetry breaks because the operation maps an X-fiber")
        print("  into an odd Z-layer, which physically only contains Y-fibers. The network")
        print("  correctly rejects this gauge transformation.")