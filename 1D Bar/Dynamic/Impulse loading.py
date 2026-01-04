import numpy as np
import scipy.linalg as la
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

start_time = time.perf_counter()

# =================================================================
# 1. INPUT PARAMETERS
# =================================================================

# Material and Geometry
E = 2e11      # Young's Modulus (N/m^2) - Steel
rho = 7850      # Density (kg/m^3) - Steel
A = 0.001       # Cross-sectional Area (m^2)
L = 1.0         # Total Length of the Bar (m)
boundary_condition = 'Fixed-Free'
# Options: 'Fixed-Free', 'Fixed-Fixed', 'Free-Free'

# FEM Discretization
Nel = 1500      # Number of Elements
Nnodes = Nel + 1
DOF = Nnodes

# Element properties
Le = L / Nel    # Length of a single element
N_ip = 2      # Options: 1, 2, or 3 (2 is typically used for exact integration)

# Forcing Parameters
F = 1000 #Magnitude of the instantaneous force (Impulse magnitude)
T_impulse = 0.000001  # Duration of the impulse (s)
F_mag = F * T_impulse    # (N/s) 
# a = 50
Node_force = Nel - 1 # Apply impulse at the free end (last node index)

# Time Integration Parameters
T_final =0.001  #time to simulation (s)
Nt = 1500 # Number of time steps
time_steps = np.linspace(0, T_final, Nt)


# Animation Parameters
animation_interval_ms = 50#Time delay between frames in milliseconds
animation_filename = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\bar_displacement_animation.mp4' # Output filename

# =================================================================
# 2. GAUSSIAN QUADRATURE SETUP
# =================================================================
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
# 3. GLOBAL ASSEMBLY 
# =================================================================

# Interpolation function to compute shape functions and their derivatives
def get_element_matrices(xi, J):
    N1, N2 = 0.5 * (1 - xi), 0.5 * (1 + xi) #Isoparametric Shape functions
    N = np.array([N1, N2])
    dN_dxi = np.array([-0.5, 0.5]) 
    B = dN_dxi / J #strain-displacement Matrix
    return N, B


# Initialize Global Matrices
K_global = np.zeros((DOF, DOF))
M_global = np.zeros((DOF, DOF))

# Assembly Loop using Numerical Integration
for e in range(Nel):
    Ke, Me = np.zeros((2, 2)), np.zeros((2, 2))
    for i in range(N_ip):
        xi, w = xi_points[i], weights[i]
        N, B = get_element_matrices(xi, J)
        Ke += (B[:, np.newaxis] @ B[np.newaxis, :]) * (E * A) * w * J
        Me += (N[:, np.newaxis] @ N[np.newaxis, :]) * (rho * A) * w * J

    n1, n2 = e, e + 1
    K_global[n1:n2+1, n1:n2+1] += Ke
    M_global[n1:n2+1, n1:n2+1] += Me

# =================================================================
# Boundary Condoitions
# =================================================================

# Dynamically determine constrained_dof based on boundary_condition
if boundary_condition == 'Fixed-Free':
    constrained_dof = [0]
elif boundary_condition == 'Fixed-Fixed':
    constrained_dof = [0, DOF - 1]
elif boundary_condition == 'Free-Free':
    constrained_dof = []
else:
    raise ValueError("Invalid boundary_condition specified. Choose from 'Fixed-Free', 'Fixed-Fixed', or 'Free-Free'.")


free_dof = np.setdiff1d(np.arange(DOF), constrained_dof)
N_free_dof = len(free_dof)

# Reduced Global Matrices (K* and M*)
K_reduced = K_global[np.ix_(free_dof, free_dof)]
M_reduced = M_global[np.ix_(free_dof, free_dof)]

print("global Stiffness Matrix:\n", K_global)
print("global Mass Matrix:\n",M_global)

F_impulse_vector = np.zeros(DOF)
F_impulse_vector[Node_force] = F_mag

print("Force Vector:\n", F_impulse_vector)

# Reduce the force vector to free DOFs
F_impulse_reduced = F_impulse_vector[free_dof]
print("Reduced Force vector:\n", F_impulse_reduced)


# =================================================================
# 4. MODAL ANALYSIS (Eigenvalue Problem)
# =================================================================

# Solve: K* * Phi = lambda * M* * Phi
eigenvalues, phi_modes = la.eigh(K_reduced, M_reduced)
omega = np.sqrt(np.real(eigenvalues)) # Natural angular frequencies
natural_frequencies = (omega) / (2 * np.pi)
print("Natural Frequencies (Hz):", natural_frequencies)

# 4a. Mass Normalization of Mode Shapes (Phi^T * M * Phi = I)
# We scale the eigenvectors such that they are M-orthogonal
M_diag = np.diag(phi_modes.T @ M_reduced @ phi_modes)
Phi_normalized = phi_modes / np.sqrt(M_diag)
print(Phi_normalized)


file_path = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\theoretical_frequencies.txt'
Matrix = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Matrix.txt'
content_to_write = "Calculated Frequencies (Hz):\n" + "\n".join([f"{omega}" for omega in natural_frequencies]) + "Global Stiffness Matrix:\n" + "\n".join([f"{K_global}" for f in K_global]) + "Global Mass Matrix:\n" + "\n".join([f"{M_global}" for f in M_global]) + "Force Vector:\n" + "\n".join([f"{F_impulse_vector}" for f in F_impulse_vector])
with open(file_path, 'w') as file:
    file.write(content_to_write)

np.savetxt(Matrix, M_global, delimiter=',')


# =================================================================
# 5. MODAL SUPERPOSITION FOR IMPULSE RESPONSE
# =================================================================

# 5a. Initial Modal Velocities (v_0)
# v_0 = Phi_normalized^T * F_impulse_reduced
# This converts the physical impulse F_impulse into modal initial velocities v_0
v0_modal = Phi_normalized.T @ F_impulse_reduced
print(v0_modal)
q0_modal = np.zeros(N_free_dof) # Initial modal displacement is zero

# 5b. Define the Modal Equation of Motion (homogeneous, for free vibration)
# q_n_ddot + omega_n^2 * q_n = 0
# State vector Y = [q, q_dot]
def modal_ode(Y, t, omega_n):
    q, q_dot = Y
    q_ddot = -omega_n**2 * q
    return [q_dot, q_ddot]

# 5c. Time Integration for EACH Mode
modal_response = np.zeros((Nt, N_free_dof))

print("Solving time integration for each mode...")
for n in range(N_free_dof):
    omega_n = omega[n]

    # Initial conditions for mode n: [q_n(0), q_dot_n(0)]
    Y0 = [q0_modal[n], v0_modal[n]]

    # Solve the simple harmonic equation
    # odeint returns Y = [q_n(t), q_dot_n(t)]
    sol = odeint(modal_ode, Y0, time_steps, args=(omega_n,))

    print(f"iteration No:{n} | f_n:{(omega_n) / (2 * np.pi)} | Max |q_n(t)|:{np.max(np.abs(sol[:,0]))} | at t={T_final}s")


    # Extract the modal displacement q_n(t)
    modal_response[:,n] = sol[:, 0]

print("Time integration complete.")
print("shape:", modal_response)


# 5d. Summation (Convert Modal Displacement q(t) back to Physical Displacement d(t))
# d(t) = Phi_normalized * q(t)
displacement_reduced = modal_response @ Phi_normalized.T
displacement_full = np.zeros((Nt, DOF))
displacement_full[:, free_dof] = displacement_reduced

print(displacement_reduced)

# 5d. Theoratical calculations
num_modes_to_display = min(5, len(natural_frequencies))
c = np.sqrt(E/rho)
if boundary_condition == 'Fixed-Free':
    # n = 1, 2, 3, ...
    n_values = np.arange(1, num_modes_to_display + 1)
    f_theory = (2 * n_values - 1) * (1 / (4 * L)) * c
elif boundary_condition == 'Fixed-Fixed':
    # n = 1, 2, 3, ...
    n_values = np.arange(1, num_modes_to_display + 1)
    f_theory = n_values * (1 / (2 * L)) * c
elif boundary_condition == 'Free-Free':
     # n = 0, 1, 2, ... (0 for rigid body, but we compare vibrating modes)
     # The first vibrating mode corresponds to n=1 in the formula (n-1),
     # which is the second mode overall.
     # We need to calculate num_modes_to_display + 1 theoretical frequencies
     # to skip the first (n=0) rigid body mode.
     n_values = np.arange(0, num_modes_to_display + 1)
     f_theory_all = n_values * (1 / (2 * L)) * c
     f_theory = f_theory_all[1:num_modes_to_display + 1] # Skip rigid body mode and take the next 'num_modes_to_display' frequencies
else:
    # This case should be caught by the earlier ValueError, but included for completeness
    f_theory = np.array([])

# print(f"Theoretical Frequencies (Hz): {f_theory}")

end_time = time.perf_counter()
print(f"Computation Time: {end_time - start_time:.4f} seconds")


# =================================================================
# 6. RESULTS VISUALIZATION
# =================================================================
# Plot displacement vs. time for the point where the force was applied
d_free_end = displacement_full[:, Node_force]
yi = 0
free_end_idx_reduced = np.where(free_dof == Node_force)[0][0]
q1_t = modal_response[:, yi] # 1st mode is at index 0
phi1_at_free_end = Phi_normalized[free_end_idx_reduced, yi]
d1_free_end = q1_t * phi1_at_free_end


file_path = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\theoretical_frequenciesMA.txt'
MatrixMA = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\MatrixMA.txt'
content_to_write = "Displacement (m): \n"+ "\n".join([f"{displacement_full}" for displacement_full in d_free_end.T]) +  "Calculated Frequencies (Hz):\n" + "\n".join([f"{omega}" for omega in natural_frequencies]) + "Global Stiffness Matrix:\n" + "\n".join([f"{K_global}" for f in K_global]) + "Global Mass Matrix:\n" + "\n".join([f"{M_global}" for f in M_global]) + "Force Vector:\n" + "\n".join([f"{F_impulse_vector}" for f in F_impulse_vector])
with open(file_path, 'w') as file:
    file.write(content_to_write)

np.savetxt(Matrix, M_global, delimiter=',')


# # Calculate wave speed for printing
c = np.sqrt(E / rho)

print("-" * 70)
print(f"Simulation of Fixed-Free Bar with (F={F_mag}N at x={L}m)")
print(f"Time: 0 to {T_final} seconds")
print("-" * 70)
print(f"Max displacement at free end: {np.max(np.abs(d_free_end))} m")
print(f"Wave speed (c): {c:.4f} m/s")
print("-" * 70)

# Find time at which the displacement magnitude at the free end is maximum
idx_max_abs = int(np.argmax(np.abs(d_free_end)))
t_max_abs = time_steps[idx_max_abs]
d_max_abs = d_free_end[idx_max_abs]
print(f"Time of max |displacement| at free end: {t_max_abs} s (disp = {d_max_abs} m)")

# 6a. Plot Displacement vs. Time at the Free End
plt.figure(figsize=(10, 5))
plt.plot(time_steps, d_free_end, label=f'Free End Displacement (Node {Node_force+1})')
# plt.plot(time_steps, d1_free_end, label=f'1st Mode Contribution', linestyle='--', color='r')
plt.title('Displacement at Free End due to Instantaneous Impulse')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid(True, linestyle='--')
plt.legend()
plt.ylim(-5e-8, 5e-8)
# plt.show()
plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\bar_displacement_time_history.jpeg', format='jpeg', dpi=600)

# 6b. Plot a snapshot of the bar's displacement shape at a specific time
snapshot_time_index = int(Nt * 0.5) # Snapshot at 50% of T_final
snapshot_time = time_steps[snapshot_time_index]

x_nodes = np.linspace(0, L, Nnodes)

plt.figure(figsize=(10, 5))
plt.plot(x_nodes, displacement_full[snapshot_time_index, :] , '-o', markersize=4)
plt.title(f'Bar Displacement Snapshot at t = {snapshot_time} s')
plt.xlabel('Position along Bar (m)')
plt.ylabel('Displacement (m)')
plt.axvline(L, color='red', linestyle='--', label='Free End')
plt.axvline(0, color='gray', linestyle='--', label='Fixed End')
plt.grid(True, linestyle='--')
# plt.show()
plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\bar_displacement_snapshot.jpeg', format='jpeg', dpi=600)


# 6c. Animation of Bar Displacement
fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
x_nodes = np.linspace(0, L, Nnodes)

# Set initial plot limits and labels
max_disp_val = np.max(np.abs(displacement_full))
ax_anim.set_xlim(0, L)
ax_anim.set_ylim(-max_disp_val, max_disp_val) # Symmetrical y-limits
ax_anim.set_xlabel('Position along Bar (m)')
ax_anim.set_ylabel('Displacement ($\mu$m)')
ax_anim.set_title(f'Bar Displacement Animation')
ax_anim.grid(True, linestyle='--')
ax_anim.axvline(L, color='red', linestyle='--', label='Free End')
ax_anim.axvline(0, color='gray', linestyle='--', label='Fixed End')
ax_anim.legend()

# Initialize the plot line for the animation
line, = ax_anim.plot([], [], 'b', markersize=4)
time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes) # Text for time

def animate_frame(frame):
    """Update the plot for each frame of the animation."""
    current_time = time_steps[frame]
    current_displacement = displacement_full[frame, :] # Convert to micrometers
    
    line.set_data(x_nodes, current_displacement)
    time_text.set_text(f't = {current_time:.2f} $\mu$s')
    
    return line, time_text

# Create the animation
print(f"Generating animation ({Nt} frames)")
ani = animation.FuncAnimation(
    fig_anim, animate_frame, frames=Nt, blit=True, interval=animation_interval_ms
)

# Save the animation file
ani.save(animation_filename, writer='ffmpeg', fps=15) # fps can be adjusted