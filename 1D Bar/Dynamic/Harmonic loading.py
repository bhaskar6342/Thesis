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
E = 2.0e11      # Young's Modulus (Pa) - Steel
rho = 7850      # Density (kg/m^3) - Steel
A = 0.001       # Cross-sectional Area (m^2)
L = 1.0         # Total Length of the Bar (m)
boundary_condition = 'Fixed-Free'
# Options: 'Fixed-Free', 'Fixed-Fixed', 'Free-Free'


# FEM Discretization
Nel = 196        # Number of Elements
Nnodes = Nel + 1
DOF = Nnodes

# Element properties
Le = L / Nel    # Length of a single element
N_ip = 2        # Options: 1, 2, or 3 (2 is typically used for exact integration)

# --- MODIFIED: Forcing Parameters ---
F_amplitude = 1000        # Force Amplitude (N)
forcing_frequency_hz = 500  # Forcing frequency (Hz)
Node_force = DOF - 1         # Apply force at the free end (last node index)
forcing_omega = 2 * np.pi * forcing_frequency_hz # Forcing angular frequency (rad/s)

# --- MODIFIED: Time Integration Parameters ---
T_final = 0.02  # Simulate for 20 milliseconds to see response
Nt = 1500       # Number of time steps
time_steps = np.linspace(0, T_final, Nt)

# # Animation Parameters (Not used in this static plot, but kept)
# animation_interval_ms = 50 
# animation_filename = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Harmonic_Response_animation.mp4' # Output filename

# =================================================================
# 2. GAUSSIAN QUADRATURE SETUP
# =================================================================
def get_gauss_data(N_ip):
    """Returns Gauss points (xi) and weights (w) for a given N_ip."""
    if N_ip == 1:
        xi = np.array([0.0])
        w = np.array([2.0])
    elif N_ip == 2:
        xi = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        w = np.array([1.0, 1.0])
    elif N_ip == 3:
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

# Jacobian (J) for 1D element
J = Le / 2

# Helper function to compute shape functions and their derivatives
def get_element_matrices(xi, J):
    N1, N2 = 0.5 * (1 - xi), 0.5 * (1 + xi)
    N = np.array([N1, N2])
    dN_dxi = np.array([-0.5, 0.5])
    B = dN_dxi / J
    return N, B

# =================================================================
# 3. GLOBAL ASSEMBLY
# =================================================================

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
# 4. BOUNDARY CONDITIONS
# =================================================================

if boundary_condition == 'Fixed-Free':
    constrained_dof = [0]
elif boundary_condition == 'Fixed-Fixed':
    constrained_dof = [0, DOF - 1]
elif boundary_condition == 'Free-Free':
    constrained_dof = []
else:
    raise ValueError("Invalid boundary_condition specified.")

free_dof = np.setdiff1d(np.arange(DOF), constrained_dof)
N_free_dof = len(free_dof)

# Reduced Global Matrices
K_reduced = K_global[np.ix_(free_dof, free_dof)]
M_reduced = M_global[np.ix_(free_dof, free_dof)]

# --- MODIFIED: Define the External Force AMPLITUDE Vector ---
F_amplitude_vector = np.zeros(DOF)
F_amplitude_vector[Node_force] = F_amplitude
F_amplitude_reduced = F_amplitude_vector[free_dof]

# =================================================================
# 5. MODAL ANALYSIS (Eigenvalue Problem)
# =================================================================

eigenvalues, phi_modes = la.eigh(K_reduced, M_reduced)
omega = np.sqrt(np.real(eigenvalues))

natural_frequencies = omega / (2 * np.pi)

# Mass Normalization of Mode Shapes
M_diag = np.diag(phi_modes.T @ M_reduced @ phi_modes)
Phi_normalized = phi_modes / np.sqrt(M_diag)

# =================================================================
# 6. MODAL SUPERPOSITION FOR HARMONIC RESPONSE (TIME DOMAIN)
# =================================================================

# Project force amplitude vector onto modal basis
P_amplitude_modal = Phi_normalized.T @ F_amplitude_reduced

# Define the FORCED Modal Equation of Motion
def modal_ode_forced(Y, t, omega_n, p_n_amp, force_omega):
    """
    Defines the forced 2nd order ODE for a single mode.
    Equation: q_ddot + (omega_n^2 * q) = p_n_amp * sin(force_omega * t)
    """
    q, q_dot = Y
    
    # Modal force at time t
    p_t = p_n_amp * np.sin(force_omega * t)
    
    # Calculate q_ddot
    q_ddot = p_t - (omega_n**2 * q)
    
    return [q_dot, q_ddot]

# Time Integration for EACH Mode
modal_response = np.zeros((Nt, N_free_dof))

print("Solving time integration for each mode...")
for n in range(N_free_dof):
    omega_n = omega[n]
    p_n_amp = P_amplitude_modal[n]
    
    # Initial conditions are [q=0, q_dot=0] (start from rest)
    Y0 = [0.0, 0.0] 
    
    # Call the new ODE solver
    sol = odeint(modal_ode_forced, Y0, time_steps, args=(omega_n, p_n_amp, forcing_omega))
    print(f"iteration No:{n} | f_n:{(omega_n) / (2 * np.pi)} | Max |q_n(t)|:{np.max(np.abs(sol[:,0]))} | at t={T_final}s")

    
    modal_response[:,n] = sol[:, 0] # Store displacement (q)
print("Time integration complete.")

# Summation to get Physical Displacement
displacement_reduced = modal_response @ Phi_normalized.T
displacement_full = np.zeros((Nt, DOF))
displacement_full[:, free_dof] = displacement_reduced

# =================================================================
# 7. THEORETICAL CALCULATIONS
# =================================================================
num_modes_to_display = min(5, len(natural_frequencies))
c = np.sqrt(E/rho)
if boundary_condition == 'Fixed-Free':
    n_values = np.arange(1, num_modes_to_display + 1)
    f_theory = (2 * n_values - 1) * c / (4 * L)
elif boundary_condition == 'Fixed-Fixed':
    n_values = np.arange(1, num_modes_to_display + 1)
    f_theory = n_values * c / (2 * L)
elif boundary_condition == 'Free-Free':
    n_values = np.arange(1, num_modes_to_display + 1)
    f_theory = n_values * c / (2 * L)

print("-" * 70)
# Print statement for harmonic force
print(f"Simulation of {boundary_condition} Bar with Harmonic Force")
print(f"Force: F(t) = {F_amplitude:.0f} * sin(2*pi*{forcing_frequency_hz:.1f}*t) N at x={L}m")
print(f"Time: 0 to {T_final:.0f} milliseconds")
print(f"Wave speed (c): {c:.1f} m/s | Travel time (L/c): {L/c:.1f} µs")
print("-" * 70)
print("Comparison of Natural Frequencies (Hz):")
print(f"{'Mode':<15} | {'FEM':<15} | {'Theoretical':<15}")
print("-" * 40)
for i in range(num_modes_to_display):
    print(f"{i+1:<5} | {natural_frequencies[i]:<15.2f} | {f_theory[i]:<15.2f}")
print("-" * 70)

# =================================================================
# 8. RESULTS VISUALIZATION (TIME DOMAIN)
# =================================================================
d_free_end = displacement_full[:, Node_force]

# Generate the force vs. time data ---
force_at_time = F_amplitude * np.sin(forcing_omega * time_steps)

# Setup dual-axis plot ---
fig, ax1 = plt.subplots(figsize=(12, 7))

fig.suptitle(f'Response at Free End (Forcing Freq = {forcing_frequency_hz} Hz)', fontsize=16)

# Plot Displacement (Left Axis)
color = 'tab:blue'
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Displacement (m)', fontsize=12, color=color)
# Use lns1 for the legend
lns1 = ax1.plot(time_steps, d_free_end, color=color, label='Displacement (m)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot Force (Right Axis)
color = 'tab:red'
ax2.set_ylabel('Force (N)', fontsize=12, color=color)
# Use lns2 for the legend
lns2 = ax2.plot(time_steps, force_at_time, color=color, linestyle='--', label='Force (N)')
ax2.tick_params(axis='y', labelcolor=color)

# --- NEW: Add a combined legend for both axes ---
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')

fig.tight_layout() # Adjust plot to prevent labels from overlapping
# plt.subplots_adjust(top=0.92) # Make space for the suptitle
plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Harmonic_Response_plot.jpeg', format='jpeg', dpi=600)
# =================================================================
file_path = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\theoretical_frequencies.txt'
Matrix = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Matrix.txt'
content_to_write = "Displacement (m): \n"+ "\n".join([f"{displacement_full}" for displacement_full in d_free_end]) +  "Calculated Frequencies (Hz):\n" + "\n".join([f"{omega}" for omega in natural_frequencies]) + "Global Stiffness Matrix:\n" + "\n".join([f"{K_global}" for f in K_global]) + "Global Mass Matrix:\n" + "\n".join([f"{M_global}" for f in M_global])
with open(file_path, 'w') as file:
    file.write(content_to_write)

np.savetxt(Matrix, M_global, delimiter=',')


# # =================================================================
# # 9. FREQUENCY RESPONSE (FRF) CALCULATION
# # =================================================================
# print("Calculating Frequency Response Function (FRF)...")

# # Define frequency sweep
# f_max = natural_frequencies[num_modes_to_display - 1] * 1.2 # 20% past 5th mode
# N_freq_steps = 1000
# freq_sweep_hz = np.linspace(1.0, f_max, N_freq_steps)
# omega_sweep = 2 * np.pi * freq_sweep_hz

# # Array to store the amplitude
# amplitude_response = np.zeros(N_freq_steps)

# # Find the index of the driving point in the REDUCED system
# # This is needed to extract the correct displacement U
# try:
#     drive_point_index_reduced = np.where(free_dof == Node_force)[0][0]
# except IndexError:
#     print(f"Error: Node_force {Node_force} is not in the free_dof list.")
#     # This would happen if you apply force to a fixed node
#     exit()

# # Loop over all frequencies
# for i, w_f in enumerate(omega_sweep):
#     # Form the dynamic stiffness matrix: D = (K - w^2 * M)
#     D_dynamic = K_reduced - (w_f**2) * M_reduced
    
#     try:
#         # Solve for the steady-state amplitude vector: U = D^-1 * F
#         U_amplitude_vector = la.solve(D_dynamic, F_amplitude_reduced)
        
#         # Get the amplitude magnitude at the driving point
#         amplitude_response[i] = np.abs(U_amplitude_vector[drive_point_index_reduced])
        
#     except la.LinAlgError:
#         # This happens if w_f is exactly on a natural frequency (singularity)
#         amplitude_response[i] = np.inf

# print("FRF calculation complete.")
# end_time = time.perf_counter()
# print(f"Computation Time: {end_time - start_time:.4f} seconds")

# # =================================================================
# # 10. FRF VISUALIZATION
# # =================================================================

# plt.figure(figsize=(12, 6))
# # Plot amplitude in mm on a log scale to see the peaks and valleys
# plt.semilogy(freq_sweep_hz, amplitude_response * 1000)

# plt.title('Frequency Response Function (FRF) at Driving Point', fontsize=16)
# plt.xlabel('Forcing Frequency (Hz)', fontsize=12)
# plt.ylabel('Steady-State Amplitude (mm) - Log Scale', fontsize=12)
# plt.grid(True, which='both', linestyle=':')

# # Add vertical lines for the natural frequencies
# for i, f_n in enumerate(natural_frequencies[:num_modes_to_display]):
#     # Only add one label to avoid clutter
#     label = f'Natural Freq. (f_n)' if i == 0 else None
#     plt.axvline(x=f_n, color='red', linestyle='--', alpha=0.7, label=label)

# # Set plot limits to make it readable
# # Find min/max values, ignoring any 'inf'
# finite_amplitudes_mm = (amplitude_response * 1000)[np.isfinite(amplitude_response)]
# if len(finite_amplitudes_mm) > 0:
#     plot_min = np.min(finite_amplitudes_mm[finite_amplitudes_mm > 0]) * 0.1
#     plot_max = np.max(finite_amplitudes_mm) * 10
#     plt.ylim(bottom=plot_min, top=plot_max)

# plt.legend()
# plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\FRF_history.jpeg', format='jpeg', dpi=600)
