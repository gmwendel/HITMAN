import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import uproot
import glob

def generate_d2d_detector():
    """Generates the idealized test grid detector."""
    positions = []
    normals = []
    
    # Bounding volume +/- 8cm, pitch 2cm
    # Centers of the 2cm bins lie at -7, -5, -3, -1, 1, 3, 5, 7
    grid_coords = np.arange(-7, 8, 2)
    z_coords = np.arange(-7, 8, 2)
    
    d = 8.0 # Half-width of the detector bounding volume
    
    for i, z in enumerate(z_coords):
        # Even layers: X-directed fibers (PMTs on +X and -X faces)
        if i % 2 == 0:
            for y in grid_coords:
                # +X Face
                positions.append([d, y, z])
                normals.append([-1.0, 0.0, 0.0])
                
                # -X Face
                positions.append([-d, y, z])
                normals.append([1.0, 0.0, 0.0])
                
        # Odd layers: Y-directed fibers (PMTs on +Y and -Y faces)
        else:
            for x in grid_coords:
                # +Y Face
                positions.append([x, d, z])
                normals.append([0.0, -1.0, 0.0])
                
                # -Y Face
                positions.append([x, -d, z])
                normals.append([0.0, 1.0, 0.0])
                
    return np.array(positions), np.array(normals)

def load_mc_detector():
    """Loads the true GEANT4 Monte Carlo geometry from ROOT."""
    files = glob.glob("/tank/BPULSE/Simulations/data/realistic_test_pt2/*.root")
    if not files:
        raise FileNotFoundError("Could not find MC ROOT files.")
        
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

    unique_pmts, indices = np.unique(positions, axis=0, return_index=True)
    positions = unique_pmts
    normals = normals[indices]
    
    return positions, normals

def plot_detector(positions, normals, prefix=""):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Heuristically separate PMTs by face for coloring
    max_x = np.max(np.abs(positions[:, 0]))
    max_y = np.max(np.abs(positions[:, 1]))
    
    x_faces = np.abs(positions[:, 0]) > max_x * 0.95
    y_faces = np.abs(positions[:, 1]) > max_y * 0.95
    
    # Scatter positions
    ax.scatter(positions[x_faces, 0], positions[x_faces, 1], positions[x_faces, 2], 
               c='b', label='X-Face PMTs', s=40, depthshade=True)
    ax.scatter(positions[y_faces, 0], positions[y_faces, 1], positions[y_faces, 2], 
               c='r', label='Y-Face PMTs', s=40, depthshade=True)
    
    # Quiver plot for inward normals (scaled for visibility)
    # Scale quiver length based on detector size
    q_len = max_x * 0.25
    ax.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
              normals[:, 0], normals[:, 1], normals[:, 2],
              length=q_len, color='k', arrow_length_ratio=0.2, alpha=0.5)
    
    # Formatting
    ax.set_xlim([-max_x*1.2, max_x*1.2])
    ax.set_ylim([-max_y*1.2, max_y*1.2])
    max_z = np.max(np.abs(positions[:, 2]))
    ax.set_zlim([-max_z*1.2, max_z*1.2])
    
    units = "mm" if prefix == "mc_" else "cm"
    ax.set_xlabel(f'X ({units})', fontweight='bold')
    ax.set_ylabel(f'Y ({units})', fontweight='bold')
    ax.set_zlabel(f'Z ({units})', fontweight='bold')
    
    title_prefix = "MC Geometry:" if prefix == "mc_" else "Ideal Grid:"
    ax.set_title(f'{title_prefix} D2d Alternating Detector ({len(positions)} Sensors)', fontweight='bold')
    ax.legend()
    
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    
    out_name = f'repos/HITMAN/symmetries/{prefix}detector_3d.png'
    plt.savefig(out_name)
    print(f"Saved {out_name}")
    plt.close()

def get_Oh_matrices():
    """Generates all 48 orthogonal matrices of the cubic group O_h."""
    matrices = []
    for p in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, p[i]] = signs[i]
            matrices.append(R)
    return matrices

def hash_lattice(pos, norm):
    combined = np.hstack([pos, norm])
    combined = np.round(combined, 1) # 1 decimal tolerance
    hashed = sorted([tuple(row) for row in combined])
    return hashed

def extract_symmetries(positions, normals):
    Oh_matrices = get_Oh_matrices()
    reference_hash = hash_lattice(positions, normals)
    
    valid_symmetries = []
    for R in Oh_matrices:
        mapped_pos = positions @ R.T
        mapped_norm = normals @ R.T
        test_hash = hash_lattice(mapped_pos, mapped_norm)
        if test_hash == reference_hash:
            valid_symmetries.append(R)
            
    return valid_symmetries

def main():
    parser = argparse.ArgumentParser(description="Explore D2d Detector Symmetries")
    parser.add_argument('--mc', action='store_true', help="Use the true MC geometry instead of the ideal grid.")
    args = parser.parse_args()

    prefix = "mc_" if args.mc else ""

    if args.mc:
        print("--- Loading True MC Geometry ---")
        positions, normals = load_mc_detector()
    else:
        print("--- Generating Ideal Test Grid ---")
        positions, normals = generate_d2d_detector()
        
    print(f"Loaded {len(positions)} PMTs.")
    
    np.save(f'repos/HITMAN/symmetries/{prefix}pmt_positions.npy', positions)
    np.save(f'repos/HITMAN/symmetries/{prefix}pmt_normals.npy', normals)

    plot_detector(positions, normals, prefix)

    # --- Symmetries ---
    symmetries = extract_symmetries(positions, normals)

    proper = [R for R in symmetries if np.linalg.det(R) > 0]
    improper = [R for R in symmetries if np.linalg.det(R) < 0]

    print(f"Total O_h matrices tested: 48")
    print(f"Total valid symmetries found: {len(symmetries)} (Expected: 8 for D_2d)\n")

    # --- Orbit Extraction ---
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
                if np.allclose(positions[j], mapped_pos, atol=1e-1) and \
                   np.allclose(normals[j], mapped_norm, atol=1e-1):
                    orbit_indices.append(j)
                    visited.add(j)
                    break
                    
        # Lexicographical Sort
        best_seed = max(orbit_indices, key=lambda idx: (positions[idx, 0], positions[idx, 2], positions[idx, 1]))
        canonical_seeds_idx.append(best_seed)

    print(f"Extracted {len(canonical_seeds_idx)} discrete symmetry orbits.")

    # --- Gram-Schmidt Frame Generation & 2D Projection ---
    latent_V = []
    latent_W = []

    global_Z = np.array([0.0, 0.0, 1.0])
    global_X = np.array([1.0, 0.0, 0.0])

    for idx in canonical_seeds_idx:
        pos = positions[idx]
        norm = normals[idx]
        
        u_seed = norm
        v_prime = np.cross(global_Z, u_seed)
        
        # If normal is aligned with Z, use X as reference
        if np.linalg.norm(v_prime) < 1e-5:
            v_prime = np.cross(global_X, u_seed)
            
        v_seed = v_prime / np.linalg.norm(v_prime)
        w_seed = np.cross(u_seed, v_seed)
        
        assert np.allclose(np.dot(np.cross(u_seed, v_seed), w_seed), 1.0), "Frame is left-handed!"
        
        V_seed = np.dot(pos, v_seed)
        W_seed = np.dot(pos, w_seed)
        
        latent_V.append(np.abs(V_seed))
        latent_W.append(W_seed)

    # --- Visualizing ---
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                c='lightgrey', s=20, alpha=0.3, label='Symmetric Aliases')

    seed_pos = positions[canonical_seeds_idx]
    ax1.scatter(seed_pos[:, 0], seed_pos[:, 1], seed_pos[:, 2], 
                c='red', s=100, marker='*', depthshade=False, label='Canonical Seeds')

    max_x = np.max(np.abs(positions[:, 0]))
    max_y = np.max(np.abs(positions[:, 1]))
    max_z = np.max(np.abs(positions[:, 2]))
    ax1.set_xlim([-max_x*1.2, max_x*1.2])
    ax1.set_ylim([-max_y*1.2, max_y*1.2])
    ax1.set_zlim([-max_z*1.2, max_z*1.2])
    
    units = "mm" if args.mc else "cm"
    ax1.set_xlabel(f'X ({units})')
    ax1.set_ylabel(f'Y ({units})')
    ax1.set_zlabel(f'Z ({units})')
    ax1.set_title('Physical Gauge Fixing: The Transversal', fontweight='bold')
    ax1.set_box_aspect([1,1,1])
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.scatter(latent_V, latent_W, c='blue', s=100, marker='s', edgecolors='k')

    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel(r'$|V_{seed}|$ (Absolute Transverse Coordinate)', fontsize=12)
    ax2.set_ylabel(r'$W_{seed}$ (Elevation Coordinate)', fontsize=12)
    ax2.set_title('The Network View: Unrolled 2D Fundamental Domain', fontweight='bold')

    plt.tight_layout()
    out_name = f'repos/HITMAN/symmetries/{prefix}latent_space.png'
    plt.savefig(out_name)
    print(f"Saved {out_name}")
    plt.close()

    U_seeds = [np.dot(positions[idx], normals[idx]) for idx in canonical_seeds_idx]
    print(f"Values of U_seed across all canonical points: {np.unique(np.round(U_seeds, 2))} {units}")

if __name__ == "__main__":
    main()
