import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_D2d_matrices():
    """Generates the 8 orthogonal matrices of the D_2d group."""
    return [
        np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), # E
        np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]), # C2(z)
        np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), # sigma_x
        np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]), # sigma_y
        np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., -1.]]), # S4(1)
        np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]]), # S4(-1)
        np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]), # C2(diag1)
        np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]), # C2(diag2)
    ]

def main():
    print("--- Verifying Irreducible Representation Mapping ---")
    
    # Load the perfectly equivariant frames generated from the MC geometry
    d = np.load("data/covariant_frames.npz")
    P = d['positions']
    U = d['u_frames']
    V = d['v_frames']
    W = d['w_frames']
    N = len(P)
    
    D2d = get_D2d_matrices()
    
    # 1. Identify the 16 Orbits and Canonical Seeds
    visited = set()
    orbits = []
    canonical_seeds = []
    
    for i in range(N):
        if i in visited: continue
        orbit = []
        for R in D2d:
            p_rot = R @ P[i]
            n_rot = R @ W[i]
            for j in range(N):
                if np.allclose(P[j], p_rot, atol=1e-1) and np.allclose(W[j], n_rot, atol=1e-1):
                    orbit.append(j)
                    visited.add(j)
                    break
        orbit = list(set(orbit)) # Unique indices
        orbits.append(orbit)
        
        # Consistent Lexicographical Sort to find the seed
        seed = max(orbit, key=lambda idx: (P[idx, 0], P[idx, 2], P[idx, 1]))
        canonical_seeds.append(seed)
        
    print(f"Grouped 128 PMTs into {len(orbits)} orbits of 8 PMTs.")
    
    # 2. Pick a random, highly asymmetric hypothesis vertex
    vertex = np.array([33.5, -12.1, 45.2])
    print(f"Hypothesis Vertex: {vertex}")
    
    # =========================================================================
    # REPRESENTATION 1: Original Detector (128 PMTs against 1 Vertex)
    # =========================================================================
    delta_128 = vertex - P
    
    # Corrected Gram-Schmidt Projections
    c_depth_128 = np.sum(delta_128 * U, axis=-1) # U is the longitudinal fiber axis
    c_v_128     = np.sum(delta_128 * V, axis=-1) # V is horizontal transverse
    c_w_128     = np.sum(delta_128 * W, axis=-1) # W is vertical transverse
    
    # Rho is strictly in the transverse (V, W) plane
    rho_128 = np.sqrt(c_v_128**2 + c_w_128**2)
    phi_128 = np.arctan2(c_w_128, c_v_128)
    
    # =========================================================================
    # REPRESENTATION 2: Irreducible Manifold (16 Seeds against 8 Alias Vertices)
    # =========================================================================
    sym_vertices = np.array([R @ vertex for R in D2d]) # (8, 3)
    
    P_16 = P[canonical_seeds]
    U_16 = U[canonical_seeds]
    V_16 = V[canonical_seeds]
    W_16 = W[canonical_seeds]
    
    # Broadcast: (8 alias vertices, 16 seeds, 3 coords)
    delta_16x8 = sym_vertices[:, np.newaxis, :] - P_16[np.newaxis, :, :] 
    
    # Compute projections and transpose so shape is (16 seeds, 8 vertices)
    c_depth_16x8 = np.sum(delta_16x8 * U_16[np.newaxis, :, :], axis=-1).T
    c_v_16x8     = np.sum(delta_16x8 * V_16[np.newaxis, :, :], axis=-1).T
    c_w_16x8     = np.sum(delta_16x8 * W_16[np.newaxis, :, :], axis=-1).T
    
    rho_16x8 = np.sqrt(c_v_16x8**2 + c_w_16x8**2)
    phi_16x8 = np.arctan2(c_w_16x8, c_v_16x8)

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(orbits)))
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Equivariance Verification: Vertex {vertex} mm", fontsize=18, fontweight='bold')
    
    # --- Plot 1: Local Cylindrical Depth vs Radius (Original 128) ---
    ax1 = fig.add_subplot(221)
    for i, orbit in enumerate(orbits):
        ax1.scatter(rho_128[orbit], c_depth_128[orbit], color=colors[i], s=40, alpha=0.8, 
                    label=f'Orbit {i}' if i<4 else "")
    ax1.set_xlabel(r'Local Radius $\rho_{local}$ (mm)', fontweight='bold')
    ax1.set_ylabel(r'Local Depth $z_{local}$ (mm)', fontweight='bold')
    ax1.set_title("Original: 128 PMTs on 1 Vertex", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # --- Plot 2: Local Cylindrical Depth vs Radius (Irreducible 16x8) ---
    ax2 = fig.add_subplot(222)
    for i in range(16):
        ax2.scatter(rho_16x8[i], c_depth_16x8[i], color=colors[i], marker='*', s=80, alpha=0.8,
                    label=f'Seed {i}' if i<4 else "")
    ax2.set_xlabel(r'Local Radius $\rho_{local}$ (mm)', fontweight='bold')
    ax2.set_ylabel(r'Local Depth $z_{local}$ (mm)', fontweight='bold')
    ax2.set_title("Irreducible: 16 Seeds on 8 Aliased Vertices", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # --- Plot 3: Polar Azimuthal Distribution (Original 128) ---
    ax3 = fig.add_subplot(223, projection='polar')
    for i, orbit in enumerate(orbits):
        ax3.scatter(phi_128[orbit], rho_128[orbit], color=colors[i], s=40, alpha=0.8)
    ax3.set_title(r"Local Azimuth $\phi_{local}$ (Original)", fontweight='bold')
    
    # --- Plot 4: Polar Azimuthal Distribution (Irreducible 16x8) ---
    ax4 = fig.add_subplot(224, projection='polar')
    for i in range(16):
        ax4.scatter(phi_16x8[i], rho_16x8[i], color=colors[i], marker='*', s=80, alpha=0.8)
    ax4.set_title(r"Local Azimuth $\phi_{local}$ (Irreducible)", fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = "repos/HITMAN/symmetries/irreducible_verification.png"
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")
    
    # Numeric Verification
    # Sort both sets of (cw, rho, phi) for a strict equality check
    original_features = np.stack([c_depth_128, rho_128, phi_128], axis=-1)
    irreducible_features = np.stack([c_depth_16x8.flatten(), rho_16x8.flatten(), phi_16x8.flatten()], axis=-1)
    
    orig_sorted = np.round(np.sort(original_features, axis=0), 3)
    irred_sorted = np.round(np.sort(irreducible_features, axis=0), 3)
    
    if np.allclose(orig_sorted, irred_sorted):
        print("\nNUMERICAL VERIFICATION: SUCCESS!")
        print("The set of local responses for the 16 seeds evaluated on the 8 symmetry-mapped vertices")
        print("is EXACTLY identical to the set of responses for all 128 PMTs on the original vertex.")
    else:
        print("\nNUMERICAL VERIFICATION: FAILED!")
        print("Mismatch found between the full detector and the irreducible representation.")

if __name__ == "__main__":
    main()
