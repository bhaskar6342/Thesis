# 1D Finite Element Analysis of a Bar under Axial Load
import numpy as np
import matplotlib.pyplot as plt
# Interpolate to nodal positions (just for visualization)
from scipy.interpolate import interp1d


E = 71700           # Young's modulus of Al7075-T6 [MPa]
A = 25            # Cross-sectional area [mm²] = 5mm x 5mm
L = 1000             # Total length of the bar [mm]
F = -200           # Applied load at the right end [N]
num_elements = 10   # Number of 1D elements
a = 4               # location of additional load
P = 300           # load to be applied

#----------------------------
# Input Problem parameters
# ---------------------------
# E = float(input('Enter youngs modulus (MPa): '))
# A = float(input('Enter the area of cross-section (mm^2): '))
# L = float(input('Length of the Bar (mm): '))
# F = float(input('Force Applied (N): '))
# num_elements = int(input('Enter the number of elements to be generated: '))
# a = float(input('Node number for additional load: '))
# P = float(input('Additional Force applied(N): '))

num_nodes = num_elements + 1
element_length = L / num_elements

# ------------------------
# Mesh and DOFs
# ------------------------
K = np.zeros((num_nodes, num_nodes))  # Global stiffness matrix
f = np.zeros(num_nodes)               # Global force vector
u = np.zeros(num_nodes)               # Displacement vector

# ------------------------
# Element stiffness matrix and assembly
# ------------------------
k_local = (E * A / element_length) * np.array([[1, -1], [-1, 1]])

for i in range(num_elements):
    K[i:i+2, i:i+2] += k_local
np.set_printoptions(linewidth=150)
print("Global Stiffness Matrix (K):\n", K)

# ------------------------
# Apply Boundary Conditions and Loads
# ------------------------
f[num_nodes-1] = F # Apply 100kN force at the right end
f[a] = P       
K_mod = K.copy()
f_mod = f.copy()

# Apply Dirichlet BC at node 0 (fixed end)
K_mod[0, :] = 0
K_mod[:, 0] = 0
K_mod[0, 0] = 1
f_mod[0] = 0
print("\nModified Global Stiffness Matrix (K_mod):\n", K_mod)

# ------------------------
# Solve System
# ------------------------
u = np.linalg.solve(K_mod, f_mod)
print('\nDisplacemnt (u):', u)

# ------------------------
# Postprocessing
# ------------------------
strain = np.zeros(num_elements)
stress = np.zeros(num_elements)
strain_energy_density = np.zeros(num_elements)

for i in range(num_elements):
    du = u[i+1] - u[i]
    strain[i] = du / element_length
    stress[i] = E * strain[i]
    strain_energy_density[i] = 0.5 * stress[i] * strain[i]

# ------------------------
# Output
# ------------------------
print("Nodal Displacements (in mm):")
print(np.round(u, 6))
print("\nElemental Stress (in MPa):")
print(np.round(stress, 3))
print("\nElemental Strain:")
print(np.round(strain, 6))
print("\nElemental Strain Energy Density (in J/mm³):")
print(np.round(strain_energy_density, 5))

# ------------------------
# Plotting
# ------------------------ 
x = np.linspace(0, L, num_elements)
node_positions = np.linspace(0, L, num_nodes)


print(f"\nElement Length: {element_length} mm")
print("\nlol", x)
# plt.figure(figsize=(5, 3))
# plt.subplot(4, 1, 1)
# plt.plot(x+element_length / 2, stress * 1, label="Stress (MPa)", marker='o')
# plt.plot(x+element_length / 2, strain, label="Strain", marker='s')
# plt.plot(x+element_length / 2, strain_energy_density, label="Strain Energy Density (J/mm³)", marker='^')
# plt.xlabel("Length along bar (mm)")
# plt.legend()
# plt.title("1D Bar FEA: Stress, Strain & Strain Energy Density")
# plt.grid(True)
# # plt.show()

# ------------------------
# 2. Interpolated (Continuous) Stress/Strain Plots
# ------------------------

# For interpolation, we'll plot values at element midpoints + interpolate between them
element_centers = np.linspace(element_length / 2, L - element_length / 2, num_elements)
node_positions = np.linspace(0, L, num_nodes)

# Interpolate strain
strain_interp = interp1d(element_centers, strain, kind='linear', fill_value='extrapolate')
strain_continuous = strain_interp(node_positions)

# Interpolate stress
stress_interp = interp1d(element_centers, stress, kind='linear', fill_value='extrapolate')
stress_continuous = stress_interp(node_positions)

# Interpolate energy
energy_interp = interp1d(element_centers, strain_energy_density, kind='linear', fill_value='extrapolate')
energy_continuous = energy_interp(node_positions)

# Plot all three
# plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(node_positions, stress_continuous * 1e-6, label='Stress (1e6 MPa)', marker='o')
plt.plot(node_positions, strain_continuous, label='Strain', marker='s')
plt.plot(node_positions, energy_continuous, label='Strain Energy Density (J/m³)', marker='^')
plt.title("Interpolated Stress, Strain, and Energy Density")
plt.xlabel("Position along bar (m)")
plt.legend()
plt.grid(True)
# plt.show()

plt.subplot(2, 1, 2)
plt.plot(node_positions, u, marker='o', linestyle='-', color='blue')
plt.title("Nodal Displacement")
plt.xlabel("Position along bar (mm)")
plt.ylabel("Displacement (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()