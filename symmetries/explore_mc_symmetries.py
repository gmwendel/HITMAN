import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import uproot
import glob

# --- PART 1: Load from ROOT ---
files = glob.glob("/tank/BPULSE/Simulations/data/realistic_test_pt2/*.root")

f = uproot.open(files[0])
meta = f["meta"]
positions = np.stack([
    meta["pmtX"].array(library="np")[0],
    meta["pmtY"].array(library="np")[0],
    meta["pmtZ"].array(library="np")[0]
], axis=1)

normals = np.stack([
    meta["pmtU"].array(library="np")[0],
    meta["pmtV"].array(library="np")[0],
    meta["pmtW"].array(library="np")[0]
], axis=1)

# Keep only unique PMTs just in case
unique_pmts, indices = np.unique(positions, axis=0, return_index=True)
positions = unique_pmts
normals = normals[indices]

print(f"Loaded {len(positions)} unique PMTs from {files[0]}")

# --- PART 2 ---
def get_D2d_matrices():
    """Generates the 8 orthogonal matrices of the D_2d group."""
    matrices = [
        np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), # E
        np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]), # C2(z)
        np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), # sigma_x
        np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]), # sigma_y
        np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., -1.]]), # S4(1)
        np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]]), # S4(-1)
        np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]), # C2(diag1)
        np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]), # C2(diag2)
    ]
    return matrices

def hash_lattice(pos, norm):
    combined = np.hstack([pos, norm])
    combined = np.round(combined, 1) 
    hashed = sorted([tuple(row) for row in combined])
    return hashed

def extract_symmetries():
    D2d_matrices = get_D2d_matrices()
    reference_hash = hash_lattice(positions, normals)
    
    valid_symmetries = []
    for R in D2d_matrices:
        mapped_pos = positions @ R.T
        mapped_norm = normals @ R.T
        test_hash = hash_lattice(mapped_pos, mapped_norm)
        if test_hash == reference_hash:
            valid_symmetries.append(R)
    return valid_symmetries

symmetries = extract_symmetries()

proper = [R for R in symmetries if np.linalg.det(R) > 0]
improper = [R for R in symmetries if np.linalg.det(R) < 0]

print(f"Total matrices tested: 8")
print(f"Total valid symmetries found: {len(symmetries)} (Expected: 8 for D_2d)\n")

# --- PART 3 ---
N = len(positions)
visited = set()
canonical_seeds_idx = []

for i in range(N):
    if i in visited:
        continue
        
    orbit_indices = []
    for R in symmetries:
        mapped_pos = R @ positions[i]
        mapped_norm = R @ normals[i]
        
        for j in range(N):
            if np.allclose(positions[j], mapped_pos, atol=1e-3) and \
               np.allclose(normals[j], mapped_norm, atol=1e-3):
                orbit_indices.append(j)
                visited.add(j)
                break
                
    best_seed = max(orbit_indices, key=lambda idx: (positions[idx, 0], positions[idx, 2], positions[idx, 1]))
    canonical_seeds_idx.append(best_seed)

print(f"Extracted {len(canonical_seeds_idx)} discrete symmetry orbits.")

latent_V = []
latent_W = []

global_Z = np.array([0.0, 0.0, 1.0])

for idx in canonical_seeds_idx:
    pos = positions[idx]
    norm = normals[idx]
    
    u_seed = norm
    
    v_prime = np.cross(global_Z, u_seed)
    if np.linalg.norm(v_prime) < 1e-5:
         global_X = np.array([1.0, 0.0, 0.0])
         v_prime = np.cross(global_X, u_seed)
         
    v_seed = v_prime / np.linalg.norm(v_prime)
    w_seed = np.cross(u_seed, v_seed)
    
    # Invariant Scalar Projections
    U_seed = np.dot(pos, u_seed) 
    V_seed = np.dot(pos, v_seed) 
    W_seed = np.dot(pos, w_seed) 
    
    latent_V.append(np.abs(V_seed))
    latent_W.append(W_seed)

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
            c='lightgrey', s=20, alpha=0.3, label='Symmetric Aliases')

seed_pos = positions[canonical_seeds_idx]
ax1.scatter(seed_pos[:, 0], seed_pos[:, 1], seed_pos[:, 2], 
            c='red', s=100, marker='*', depthshade=False, label='Canonical Seeds')

ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
ax1.set_title('Physical Gauge Fixing: MC Geometry', fontweight='bold')
ax1.set_box_aspect([1,1,1])
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.scatter(latent_V, latent_W, c='blue', s=100, marker='s', edgecolors='k')

ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlabel(r'$|V_{seed}|$ (Absolute Transverse Coordinate)', fontsize=12)
ax2.set_ylabel(r'$W_{seed}$ (Elevation Coordinate)', fontsize=12)
ax2.set_title('Unrolled 2D Fundamental Domain', fontweight='bold')

plt.tight_layout()
plt.savefig('/tank/BPULSE/Simulations/repos/HITMAN/symmetries/mc_latent_space.png')
print("Saved mc_latent_space.png")

U_seeds = [np.dot(positions[idx], normals[idx]) for idx in canonical_seeds_idx]
print(f"Values of U_seed across all canonical points: {np.unique(np.round(U_seeds, 2))} mm")