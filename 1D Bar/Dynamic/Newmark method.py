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

# FEM Discretization
Nel = 150      # Number of Elements
Nnodes = Nel + 1
DOF = Nnodes

# Element properties
Le = L / Nel    # Length of a single element
N_ip = 2      # Options: 1, 2, or 3 (2 is typically used for exact integration)

# Forcing Parameters
boundary_condition = 'Fixed-Free'
# Options: 'Fixed-Free', 'Fixed-Fixed', 'Free-Free'
forcing = 'sin'  # Options: 'impulse' or 'sin'
F_mag =1000    # (N) Magnitude of the force
T_impulse = 0.0000001  # Duration of the impulse (s)
forcing_frequency_Hz = 500 # Frequency for sinusoidal forcing (Hz)
forcing_frequency = 2 * np.pi * forcing_frequency_Hz  # Convert to rad/s
Node_force = Nel-1 # Apply impulse at the free end (last node index)

# Time Integration Parameters
beta = 0.25   # Newmark-beta parameter (0.25 for average acceleration)
gamma = 0.5   # Newmark-gamma parameter (0.5 for average acceleration)
T_final =0.02  #time to simulation (s)
Nt = 15000 # Number of time steps
delT = T_final / (Nt - 1) # Time step size
time_steps = np.linspace(0, T_final, Nt)

# Animation Parameters
animation_interval_ms = 50#Time delay between frames in milliseconds
if forcing == 'impulse':
    animation_filename = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Impulse\bar_displacement_animation.mp4' # Output filename
elif forcing == 'sin':
    animation_filename = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Sin\bar_displacement_animation.mp4' # Output filename
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

# Helper function to compute shape functions and their derivatives
def get_element_matrices(xi, J):
    N1, N2 = 0.5 * (1 - xi), 0.5 * (1 + xi) #Shape functions
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

# ==================================================================
# Damping Matrix (C) -> UNDAMPED SYSTEM (C = 0)
# ==================================================================
# Rayleigh Damping: C = alpha * M + beta * K
# Set alpha and beta to zero for an undamped system.
alpha = 0  # Mass proportional damping constant (Zero)
beta_d = 0 # Stiffness proportional damping constant (Zero)
C_reduced = alpha * M_reduced + beta_d * K_reduced


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

print("Mass Normalized Mode Shapes (Phi^T * M * Phi = I):\n", M_diag)
print("reduced mass matrix:\n", M_reduced)


# =================================================================
# 5. MODAL SUPERPOSITION FOR IMPULSE RESPONSE
# =================================================================

# 5a. Initial Modal Velocities (v_0)
v0_modal = np.zeros(N_free_dof)
q0_modal = np.zeros(N_free_dof) # Initial modal displacement is zero


# =================================================================
# 5b. Newmark-beta Method Setup
# =================================================================

def newmark_beta_mdof_solver(M, C, K, dt, total_time, u0, u_dot0, force_function, beta=(0.25), gamma=0.6):
    """
    Implements the Newmark-Beta method to solve the MDOF dynamic equation of motion.

    The method solves: M*ü + C*u̇ + K*u = F(t)

    Args:
        M (np.ndarray): Mass Matrix (N x N).
        C (np.ndarray): Damping Matrix (N x N).
        K (np.ndarray): Stiffness Matrix (N x N).
        dt (float): Time step size (Δt).
        total_time (float): Total duration of the simulation.
        u0 (np.ndarray): Initial displacement vector (N).
        u_dot0 (np.ndarray): Initial velocity vector (N).
        force_function (function): Function F(t) returning the external force vector F_t.
        beta (float, optional): Newmark Beta parameter. Default is 0.25.
        gamma (float, optional): Newmark Gamma parameter. Default is 0.5.

    Returns:
        tuple: (time_array, displacement_matrix, velocity_matrix, acceleration_matrix)
    """

    N = M.shape[0]  # Number of degrees of freedom
    num_steps = int(total_time / dt) + 1
    time = np.linspace(0, total_time, num_steps)

    # Output matrices where each row is a time step, and columns are DOFs
    u = np.zeros((num_steps, N))
    u_dot = np.zeros((num_steps, N))
    u_ddot = np.zeros((num_steps, N))

    # Apply Initial Conditions (must be vectors)
    u[0, :] = u0
    u_dot[0, :] = u_dot0

    # 1. Calculate Initial Acceleration (ü₀)
    # M*ü₀ + C*u̇₀ + K*u₀ = F₀  =>  M*ü₀ = F₀ - C*u̇₀ - K*u₀
    F0 = force_function(time[0])
    RHS0 = F0 - C @ u_dot[0, :] - K @ u[0, :]
    u_ddot[0, :] = np.linalg.solve(M, RHS0) # Solve M*ü₀ = RHS₀

    # 2. Calculate Newmark Constants (same scalars as before)
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt) #damping term
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0 #damping term
    a5 = dt / 2.0 * (gamma / beta - 2.0*gamma) #damping term
    a6 = dt * (1.0 - gamma)
    a7 = gamma * dt

    # 3. Calculate Effective Stiffness Matrix (K̂)
    # K̂ = K + C * a1 + M * a0
    # Note: The * operator performs scalar multiplication on the matrix here.
    K_hat = K + C * a1 + M * a0

    # 4. Time-Stepping Loop
    for i in range(num_steps - 1):
        # Retrieve state vectors at time n (row i)
        u_n = u[i, :]
        u_dot_n = u_dot[i, :]
        u_ddot_n = u_ddot[i, :]
        print(f"timestep: {time[i]}, disp: {np.max(np.abs(u_n[0:]))}")
        # 4a. Calculate the Effective Force Vector (R̂)
        # R̂ = F_{n+1} + M * (a0*u_n + a2*u̇_n + a3*ü_n) + C * (a1*u_n + a4*u̇_n + a5*ü_n)

        F_i_plus_1 = force_function(time[i+1])

        # M-term and C-term are vectors resulting from matrix-vector multiplication (@)
        M_term_vec = M @ (a0 * u_n + a2 * u_dot_n + a3 * u_ddot_n)
        C_term_vec = C @ (a1 * u_n + a4 * u_dot_n + a5 * u_ddot_n)

        R_hat = F_i_plus_1 + M_term_vec + C_term_vec

        # 4b. Solve for the unknown displacement vector at n+1 (u_{n+1})
        # K̂ * u_{n+1} = R̂
        # This is the key difference: replacing division with linear system solving
        u[i+1, :] = np.linalg.solve(K_hat, R_hat)

        # 4c. Update Acceleration and Velocity at n+1

        # ü_{n+1} = a0 * (u_{n+1} - u_n) - a2 * u̇_n - a3 * ü_n
        u_ddot[i+1, :] = a0 * (u[i+1, :] - u_n) - a2 * u_dot_n - a3 * u_ddot_n

        # u̇_{n+1} = u̇_n + a6 * ü_n + a7 * ü_{n+1}
        u_dot[i+1, :] = u_dot_n + a6 * u_ddot_n + a7 * u_ddot[i+1, :]

    return time, u, u_dot, u_ddot

print("Damping Matrix:\n", C_reduced)

# =================================================================
# Force Function for Impulse
# =================================================================
def impulse_force(t):
    """A short impulse force applied at the defined node."""
    F = np.zeros(N_free_dof)
    if t <= T_impulse:
        F[Node_force] = F_mag # Apply force to the last DOF
    return F

def Sin_force(t): 
    F = np.zeros(N_free_dof) 
    F[Node_force] = F_mag * np.sin(forcing_frequency * t) 
    return F

# def Sin_force(t): 
#     F = np.zeros(N_free_dof) 
    
#     # robustly find the index in the reduced vector that matches the global Node_force
#     # valid only if Node_force is actually free.
#     if Node_force in free_dof:
#         idx_reduced = np.where(free_dof == Node_force)[0][0]
#         F[idx_reduced] = F_mag * np.sin(forcing_frequency * t) 
        
# return F

if forcing == 'impulse':
    force=impulse_force
elif forcing == 'sin':
    force=Sin_force
else:
    raise ValueError("Invalid forcing type specified. Choose 'impulse' or 'sinusoidal'.")

# =================================================================
# 5b. Newmark-beta Time Integration for Each Mode
# =================================================================

time_steps, disp_matrix, vel_matrix, accel_matrix = newmark_beta_mdof_solver(
    M=M_reduced,
    C=C_reduced,
    K=K_reduced,
    dt=delT,
    total_time=T_final,
    u0=v0_modal,  # Initial modal displacement is zero
    u_dot0=q0_modal,  # Initial modal velocity
    force_function=force
)        


print("displacement shape:", disp_matrix.shape)


displacement_full = np.zeros((Nt, DOF))
displacement_full[:, free_dof] = disp_matrix



end_time = time.perf_counter()
print(f"Computation Time: {end_time - start_time:.4f} seconds")


# =================================================================
# 6. RESULTS VISUALIZATION
# =================================================================
# # Plot displacement vs. time for the point where the force was applied
d_free_end = displacement_full[:, Node_force]
yi = 0
free_end_idx_reduced = np.where(free_dof == Node_force)[0][0]
q1_t = disp_matrix[:, yi] # 1st mode is at index 0
phi1_at_free_end = Phi_normalized[free_end_idx_reduced, yi]
d1_free_end = q1_t * phi1_at_free_end

print("displacement at free end:", d_free_end)
if forcing == 'impulse':
    file_path = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Impulse\theoretical_frequencies.txt'
    Matrix = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Impulse\Matrix.txt'
elif forcing == 'sin':
    file_path = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Sin\theoretical_frequencies.txt'
    Matrix = r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Sin\Matrix.txt'
content_to_write = "Displacement (m): \n"+ "\n".join([f"{displacement_full}" for displacement_full in d_free_end.T]) +  "Calculated Frequencies (Hz):\n" + "\n".join([f"{omega}" for omega in natural_frequencies]) + "Global Stiffness Matrix:\n" + "\n".join([f"{K_global}" for f in K_global]) + "Global Mass Matrix:\n" + "\n".join([f"{M_global}" for f in M_global]) + "Force Vector:\n" + "\n".join([f"{F_impulse_vector}" for f in F_impulse_vector])
with open(file_path, 'w') as file:
    file.write(content_to_write)

np.savetxt(Matrix, M_global, delimiter=',')

# # Calculate wave speed for printing
c = np.sqrt(E / rho)

print("-" * 100)
print(f"Simulation of Fixed-Free Bar with (F={F_mag}N at x={L}m)")
print(f"Time: 0 to {T_final} seconds")
print("-" * 100)
print(f"Max displacement at free end: {np.max(np.abs(d_free_end))} m")
print(f"Wave speed (c): {c:.1f} m/s")
print("-" * 100)

# Find time at which the displacement magnitude at the free end is maximum
idx_max_abs = int(np.argmax(np.abs(d_free_end)))
t_max_abs = time_steps[idx_max_abs]
d_max_abs = d_free_end[idx_max_abs]
print(f"Time of max |displacement| at free end: {t_max_abs} s (disp = {d_max_abs} m)")

# 6a. Plot Displacement vs. Time at the Free End
plt.figure(figsize=(10, 5))
plt.plot(time_steps, d_free_end, label=f'Free End Displacement (Node {Node_force})')
# plt.plot(time_steps, d1_free_end, label=f'1st Mode Contribution', linestyle='--', color='r')
plt.title('Displacement at Free End due to loading')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid(True, linestyle='--')
plt.legend()
# plt.ylim(-5e-8, 5e-8)
# plt.show()
if forcing == 'impulse':
    plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Impulse\bar_displacement_time_history.jpeg', format='jpeg', dpi=600)
elif forcing == 'sin':
    plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Sin\bar_displacement_time_history.jpeg', format='jpeg', dpi=600)

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
if forcing == 'impulse':
    plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Impulse\bar_displacement_snapshot.jpeg', format='jpeg', dpi=600)
elif forcing == 'sin':
    plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\Newmark\Sin\bar_displacement_snapshot.jpeg', format='jpeg', dpi=600)

# 6c. Animation of Bar Displacement
fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
x_nodes = np.linspace(0, L, Nnodes)

# Set initial plot limits and labels
max_disp_val = np.max(np.abs(displacement_full)) * 1.1 # Slightly larger for padding
ax_anim.set_xlim(0, L)
ax_anim.set_ylim(-max_disp_val, max_disp_val) # Symmetrical y-limits
ax_anim.set_xlabel('Position along Bar (m)')
ax_anim.set_ylabel('Displacement (m)')
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
    time_text.set_text(f't = {current_time*1e6:.2f} $\mu$s')
    
    return line, time_text

# Create the animation
print(f"Generating animation ({Nt} frames)")
ani = animation.FuncAnimation(
    fig_anim, animate_frame, frames=Nt, blit=True, interval=animation_interval_ms
)

# Save the animation file
ani.save(animation_filename, writer='ffmpeg', fps=15) # fps can be adjusted