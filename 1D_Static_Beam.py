import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
np.set_printoptions(threshold=np.inf, linewidth=200)


#-------------------------------------------------------------------------------------------------------------------------
# --- Input Problem Parameters ---
#-------------------------------------------------------------------------------------------------------------------------
# E = float(input('Enter Young\'s Modulus (Pa): '))
# I = float(input('Enter Area Moment of Inertia (m^4): '))
# L = float(input('Enter the length of the beam (m): '))
# W = float(input('Enter the distributed load (N/m): '))
# P = float(input('Enter the point load (N): '))
# a = float(input('Enter the location of the point load (m): '))
# num_elements = int(input('Enter the number of elements to be generated: '))
# support = input('Enter the type of support (F, FF, PP, PR, RR): ')

E = 200e9  # Young's Modulus for steel [Pa]
I = 8.33334e-6   # Area Moment of Inertia [m^4] (e.g., for a rectangular cross-section)
L = 10      # Total length of the beam [m]
W = 000   # Uniformly distributed load [N/m]
P = 6000   # Point load [N]


#-------------------------------------------------------------------------------------------------------------------------
# --- Define supports ---
#-------------------------------------------------------------------------------------------------------------------------
# F = Cantoilever beam: fixed at left end, free at right end
# FF = Fixed at both ends
# PP = Pin supported at both ends
# PR = Pin at left end, Roller at right end
# RR = Roller at both ends
support = 'f'  # Change this to 'F', 'FF','FP', 'PP', 'PR', or 'RR' as needed


a = 7     # Location of the point load [m]
num_elements = 1000 # Number of 1D beam elements

num_nodes = num_elements + 1
element_length = L / num_elements
# Each node has 2 degrees of freedom: vertical displacement (v) and rotation (theta)
# The total number of degrees of freedom is 2 * num_nodes
num_dofs = 2 * num_nodes

#-------------------------------------------------------------------------------------------------------------------------
# --- Mesh and Global Matrices ---
#-------------------------------------------------------------------------------------------------------------------------

K = np.zeros((num_dofs, num_dofs))  # Global stiffness matrix
f = np.zeros(num_dofs)              # Global force vector

#-------------------------------------------------------------------------------------------------------------------------
# --- Element Stiffness Matrix and Assembly (built from scratch using shape functions) ---
#-------------------------------------------------------------------------------------------------------------------------

def build_element_stiffness_matrix(E, I, L_e):
    """
    Builds the element stiffness matrix for a 1D Euler-Bernoulli beam element.
    This is derived from the beam's governing differential equation and shape functions.
    The formula is: integral(EI * v'' * w'') dx
    
    Args:
        E (float): Young's Modulus.
        I (float): Area Moment of Inertia.
        L_e (float): Element length.
    
    Returns:
        np.array: The n*n element stiffness matrix.
    """
    k = np.zeros((4, 4))
    
    # Shape functions (Hermite cubic polynomials)
    # N1(x) = 1 - 3*(x/L_e)^2 + 2*(x/L_e)^3
    # N2(x) = x - 2*(x^2/L_e) + (x^3/L_e^2)
    # N3(x) = 3*(x/L_e)^2 - 2*(x/L_e)^3
    # N4(x) = -(x^2/L_e) + (x^3/L_e^2)
    
    # Second derivatives of the shape functions
    # N1''(x) = -6/L_e^2 + 12*x/L_e^3
    # N2''(x) = -4/L_e + 6*x/L_e^2
    # N3''(x) = 6/L_e^2 - 12*x/L_e^3
    # N4''(x) = -2/L_e + 6*x/L_e^2
    
    # Integrals of the product of second derivatives of shape functions
    # k_ij = integral from 0 to L_e of (E * I * N_i'' * N_j'') dx
    
    # k_11 = EI * integral(N1''*N1'') dx = EI * integral((-6/L_e^2 + 12x/L_e^3)^2) dx = 12*EI/L_e^3
    k[0, 0] = 12
    # k_12 = k_21 = EI * integral(N1''*N2'') dx = EI * integral((-6/L_e^2 + 12x/L_e^3)*(-4/L_e + 6x/L_e^2)) dx = 6*EI/L_e^2
    k[0, 1] = 6 * L_e
    k[1, 0] = 6 * L_e
    # k_13 = k_31 = EI * integral(N1''*N3'') dx = EI * integral((-6/L_e^2 + 12x/L_e^3)*(6/L_e^2 - 12x/L_e^3)) dx = -12*EI/L_e^3
    k[0, 2] = -12
    k[2, 0] = -12
    # k_14 = k_41 = EI * integral(N1''*N4'') dx = EI * integral((-6/L_e^2 + 12x/L_e^3)*(-2/L_e + 6x/L_e^2)) dx = 6*EI/L_e^2
    k[0, 3] = 6 * L_e
    k[3, 0] = 6 * L_e
    
    # k_22 = EI * integral(N2''*N2'') dx = EI * integral((-4/L_e + 6x/L_e^2)^2) dx = 4*EI/L_e
    k[1, 1] = 4 * L_e**2
    # k_23 = k_32 = EI * integral(N2''*N3'') dx = EI * integral((-4/L_e + 6x/L_e^2)*(6/L_e^2 - 12x/L_e^3)) dx = -6*EI/L_e^2
    k[1, 2] = -6 * L_e
    k[2, 1] = -6 * L_e
    # k_24 = k_42 = EI * integral(N2''*N4'') dx = EI * integral((-4/L_e + 6x/L_e^2)*(-2/L_e + 6x/L_e^2)) dx = 2*EI/L_e
    k[1, 3] = 2 * L_e**2
    k[3, 1] = 2 * L_e**2
    
    # k_33 = EI * integral(N3''*N3'') dx = EI * integral((6/L_e^2 - 12x/L_e^3)^2) dx = 12*EI/L_e^3
    k[2, 2] = 12
    # k_34 = k_43 = EI * integral(N3''*N4'') dx = EI * integral((6/L_e^2 - 12x/L_e^3)*(-2/L_e + 6x/L_e^2)) dx = -6*EI/L_e^2
    k[2, 3] = -6 * L_e
    k[3, 2] = -6 * L_e

    # k_44 = EI * integral(N4''*N4'') dx = EI * integral((-2/L_e + 6x/L_e^2)^2) dx = 4*EI/L_e
    k[3, 3] = 4 * L_e**2
    
    return (E * I / L_e**3) * k

k_local = build_element_stiffness_matrix(E, I, element_length)

for i in range(num_elements):
    
    # Map local DOFs to global DOFs for the current element
    dof_map = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
    
    # Add the element stiffness matrix to the global matrix
    for row in range(4):
        for col in range(4):
            K[dof_map[row], dof_map[col]] += k_local[row, col]

# print("Global Stiffness Matrix (K):\n", np.round(K, 1))

#-------------------------------------------------------------------------------------------------------------------------
# --- Apply Loads ---
#-------------------------------------------------------------------------------------------------------------------------

# Load vector from a uniformly distributed load (for each element)
# This is derived by integrating the product of the distributed load w(x) and the shape functions N_i(x)
# f_i = integral(w * N_i) dx
# f_1 = integral(w * N1) dx = w*L_e/2
# f_2 = integral(w * N2) dx = w*L_e^2/12
# f_3 = integral(w * N3) dx = w*L_e/2
# f_4 = integral(w * N4) dx = -w*L_e^2/12
f_distributed_local = (W * element_length / 12) * np.array([6, element_length, 6, -element_length])

for i in range(num_elements):
    dof_map = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
    f[dof_map] += f_distributed_local

# Add the point load at the nearest node
node_load_index = int(round(a / element_length))
# The displacement DOF is 2*node_index
f[2 * node_load_index] += P

print("\nGlobal Force Vector (f):\n", np.round(f, 2))

#-------------------------------------------------------------------------------------------------------------------------
# ---Improved Apply Boundary Conditions ---
#-------------------------------------------------------------------------------------------------------------------------
# The boundary conditions depend on the type of support specified.
# We will define the boundary conditions based on the 'support' variable.

support = support.upper()  # Ensure the support type is in uppercase for consistency
if support == 'F':  # Cantilever beam: fixed at left end, free at right end
    fixed_dofs = [0, 1]  # Fixed at left end (v=0, theta=0)
elif support == 'FF':  # Fixed at both ends
    fixed_dofs = [0, 1, 2*(num_nodes-1), 2*(num_nodes-1)+1]  # Fixed at left end and right end
elif support == 'FP':  # Fixed at left end, Pin at right end
    fixed_dofs = [0, 1, 2*(num_nodes-1)]  # Fixed at left end and vertical at right end
elif support == 'PP':  # Pin supported at both ends
    fixed_dofs = [0, 2*(num_nodes-1)]  # Vertical
elif support == 'PR':  # Pin at left end, Roller at right end
    fixed_dofs = [0, 2*(num_nodes-1)]  # Vertical at left end and vertical at right end
elif support == 'RR':  # Roller at both ends
    fixed_dofs = [0, 2*(num_nodes-1)]  # Vertical at left end and vertical at right end
else:
    raise ValueError(
        """Invalid support type. Choose from:
        F  = Cantilever beam: fixed at left end, free at right end
        FF = Fixed at both ends
        PP = Pin supported at both ends
        PR = Pin at left end, Roller at right end
        RR = Roller at both ends"""
    )


#-------------------------------------------------------------------------------------------------------------------------
# --- Apply Boundary Conditions ---
#-------------------------------------------------------------------------------------------------------------------------

# Cantilever beam: fixed end at x=0.
# We are fixing both the vertical displacement (v) and the rotation (theta) at node 0.
# This corresponds to DOFs 0 and 1.
# fixed_dofs = [0, 1]
# fixed_dofs = [0, 1, 2*(num_nodes-1), 2*(num_nodes-1)+1]  # Fixed at left end and right end

#-------------------------------------------------------------------------------------------------------------------------
# --- Modifying Stiffness matrix and force vector ---
#-------------------------------------------------------------------------------------------------------------------------

# Create copies of K and f to modify
# Modfiying the global stiffness matrix and force vector to apply boundary conditions
K_mod = K.copy()
f_mod = f.copy()

for dof in fixed_dofs:
    K_mod[dof, :] = 0
    K_mod[:, dof] = 0
    K_mod[dof, dof] = 1
    f_mod[dof] = 0

#-------------------------------------------------------------------------------------------------------------------------
# --- Solve System ---
#-------------------------------------------------------------------------------------------------------------------------

# Solving for the unknown displacements and rotations
u = np.linalg.solve(K_mod, f_mod)
u_v = u[::2]  # Vertical displacements (at even indices)
u_theta = u[1::2] # Rotations (at odd indices)


#-------------------------------------------------------------------------------------------------------------------------
# --- Postprocessing ---
#-------------------------------------------------------------------------------------------------------------------------

# Calculate internal forces (shear force and bending moment at node points)
node_positions = np.linspace(0, L, num_elements + 1)
shear_force = np.zeros(node_positions.shape)
bending_moment = np.zeros(node_positions.shape)

for i in range(node_positions.shape[0] - 1):
    dof_map = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
    u_element = u[dof_map]
    
    # Calculate local internal forces
    f_local_internal = k_local @ u_element
    
    # Shear force at the start and end of the element
    shear_force[i] = f_local_internal[0]
    # Bending moment at the start and end of the element
    bending_moment[i] = -f_local_internal[1]

print(u_element)

#-------------------------------------------------------------------------------------------------------------------------
# --- Output ---
#-------------------------------------------------------------------------------------------------------------------------

print("\nNodal Displacements (m):")
print(np.round(u_v, 8))
print("\nNodal Rotations (rad):")
print(np.round(u_theta, 8))
print("\nElemental Shear Force (N):")
print(np.round(shear_force, 2))
print("\nElemental Bending Moment (Nm):")
print(np.round(bending_moment, 2))

#-------------------------------------------------------------------------------------------------------------------------
# --- Plotting ---
#-------------------------------------------------------------------------------------------------------------------------

element_centers = np.linspace(element_length / 2, L - element_length / 2, num_elements)

#-------------------------------------------------------------------------------------------------------------------------
# Plotting nodal displacements
#-------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(node_positions, u_v * 1e3, marker='', linestyle='-')
plt.plot(0, 0, marker='x', color='red')
plt.plot(2.5, 474.6, marker='x', color='red')
plt.plot(5,1593.7 , marker='x', color='red')
plt.plot(7.5, 3005.83, marker='x', color='red')
plt.plot(10, 4499, marker='x', color='red')
plt.title('Nodal Vertical Displacement')
plt.xlabel('Position along beam (m)')
plt.ylabel('Displacement (mm)')
plt.legend(['Numerical', 'Exact & Analytical'])
plt.grid(True)

# Plotting shear force
plt.subplot(3, 1, 2)
plt.step(np.repeat(node_positions, 2), np.repeat(shear_force, 2), color='blue', linestyle='-')
plt.plot(0, -60000, marker='x', color='red')
plt.plot(2.5, -45000, marker='x', color='red')
plt.plot(5,-30000 , marker='x', color='red')
plt.plot(7.5, -15000, marker='x', color='red')
plt.plot(10, 0, marker='x', color='red')
plt.title('Shear Force Diagram')
plt.xlabel('Position along beam (m)')
plt.ylabel('Shear Force (N)')
plt.legend(['Numerical', 'Exact & Analytical'])
plt.grid(True)

# Plotting bending moment
plt.subplot(3, 1, 3)
plt.step(np.repeat(node_positions, 2), np.repeat(bending_moment, 2), color='green', linestyle='-')
plt.plot(0, 300000, marker='x', color='red')
plt.plot(2.5, 168750, marker='x', color='red')
plt.plot(5,75000 , marker='x', color='red')
plt.plot(7.5, 18750, marker='x', color='red')
plt.plot(10, 0, marker='x', color='red')
plt.title('Bending Moment Diagram')
plt.xlabel('Position along beam (m)')
plt.ylabel('Bending Moment (Nm)')
plt.legend(['Numerical', 'Exact & Analytical'])

plt.grid(True)

plt.tight_layout()
plt.show()
