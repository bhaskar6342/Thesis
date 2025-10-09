import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()
# =================================================================
# 1. INPUT PARAMETERS
# =================================================================

# Material and Geometry
E = 2.0e11      # Young's Modulus (Pa)
rho = 7850      # Density (kg/m^3)
A = 0.001       # Cross-sectional Area (m^2)
L = 10.0         # Total Length of the Bar (m)

# FEM Discretization
Nel = 100        # Number of Elements
Nnodes = Nel + 1
DOF = Nnodes

# Element properties
Le = L / Nel    # Length of a single element

# =================================================================
# 2. GAUSSIAN QUADRATURE SETUP (User-Defined N_ip)
# =================================================================

# --> DEFINE NUMBER OF INTEGRATION POINTS HERE <--
N_ip = 3  # Options: 1, 2, or 3 (2 is typically used for exact integration)

def get_gauss_data(N_ip):
    """Returns Gauss points (xi) and weights (w) for a given N_ip."""
    if N_ip == 1:
        # Integrates polynomial up to degree 1
        xi = np.array([0.0])
        w = np.array([2.0])
    elif N_ip == 2:
        # Integrates polynomial up to degree 3 (Exact for consistent Mass Matrix)
        xi = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        w = np.array([1.0, 1.0])
    elif N_ip == 3:
        # Integrates polynomial up to degree 5 (Higher order for future use)
        xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        w = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("N_ip must be 1, 2, or 3 for this simplified code.")
    return xi, w

try:
    xi_points, weights = get_gauss_data(N_ip)
except ValueError as e:
    print(f"Error: {e}")
    exit()

# Jacobian (J) for 1D element: dx = J * d(xi)
J = Le / 2

# =================================================================
# 3. GLOBAL ASSEMBLY (using Numerical Integration)
# =================================================================

# Function to get Shape Functions N(xi) and Strain-Displacement B
def get_element_matrices(xi):
    """Calculates N and B matrices at a given integration point xi."""
    
    # Shape Functions N (1x2 vector)
    N1 = 0.5 * (1 - xi)
    N2 = 0.5 * (1 + xi)
    N = np.array([N1, N2])
    
    # Derivative of Shape Functions w.r.t. xi
    dN_dxi = np.array([-0.5, 0.5])
    
    # Strain-Displacement Matrix B (1x2 vector)
    B = dN_dxi / J
    
    return N, B

# Initialize Global Matrices
K_global = np.zeros((DOF, DOF))
M_global = np.zeros((DOF, DOF))

# Assembly Loop
for e in range(Nel):
    Ke = np.zeros((2, 2))
    Me = np.zeros((2, 2))

    # --- Numerical Integration Loop (over Gauss points) ---
    for i in range(N_ip):
        xi = xi_points[i]
        w = weights[i]

        # Get the matrices at the current Gauss point
        N, B = get_element_matrices(xi)

        # Integrate Stiffness Matrix: Ke += (B^T * E * A * B) * w * J
        Ke += (B[:, np.newaxis] @ B[np.newaxis, :]) * (E * A) * w * J

        # Integrate Mass Matrix: Me += (N^T * rho * A * N) * w * J
        Me += (N[:, np.newaxis] @ N[np.newaxis, :]) * (rho * A) * w * J
        
    # Global DOFs for the current element
    n1 = e
    n2 = e + 1

    # Assembly
    K_global[n1:n2+1, n1:n2+1] += Ke
    M_global[n1:n2+1, n1:n2+1] += Me


print(Me, "\n", M_global)

# =================================================================
# 4. BOUNDARY CONDITIONS (Fixed-Free Bar)
# =================================================================

# Apply Fixed-Free BC: Node 0 (x=0) is fixed
constrained_dof = [0]
free_dof = np.setdiff1d(np.arange(DOF), constrained_dof)

# Reduced Global Matrices (K* and M*)
K_reduced = K_global[np.ix_(free_dof, free_dof)]
M_reduced = M_global[np.ix_(free_dof, free_dof)]

# =================================================================
# 5. EIGENVALUE PROBLEM SOLUTION
# =================================================================

eigenvalues, eigenvectors_reduced = la.eigh(K_reduced, M_reduced)
omega_squared = np.real(eigenvalues)
natural_frequencies = np.sqrt(omega_squared) / (2 * np.pi)

end_time = time.perf_counter()
print(f"Computation Time: {end_time - start_time:.4f} seconds")
# =================================================================
# 6. RESULTS AND PLOTTING
# =================================================================

print("="*70)
print(f"Dynamic FEM (Gaussian Quadrature) of a 1D Bar (Nel={Nel})")
print(f"Integration Points (N_ip): {N_ip}")
print("="*70)

# Display Frequencies
print(f"{'Mode':<5} | {'Natural Frequency (Hz)':<25}")
print("-" * 70)
for i in range(5):
    print(f"{i+1:<5} | {natural_frequencies[i]:<25.4f}")
print("="*70)

# Theoretical Frequencies for comparison
c = np.sqrt(E/rho)
f_theory = (2 * np.arange(1, 6) - 1) * (1 / (4 * L)) * c
print(f"Theoretical Frequencies (Hz): {f_theory}")

# Plotting Mode Shapes
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x_nodes = np.linspace(0, L, Nnodes)

for i in range(3):
    mode_shape = np.insert(eigenvectors_reduced[:, i], 0, 0)
    mode_shape = mode_shape / np.max(np.abs(mode_shape))
    
    ax = axes[i]
    ax.plot(x_nodes, mode_shape, '-o', markersize=4)
    ax.set_title(f'Mode {i+1} (f = {natural_frequencies[i]:.2f} Hz)')
    ax.set_xlabel('Position x (m)')

plt.tight_layout()
plt.show()