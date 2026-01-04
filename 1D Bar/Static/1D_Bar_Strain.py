import numpy as np
import matplotlib.pyplot as plt

E = 71700           # Young's modulus of Al7075-T6 [MPa]
A = 25            # Cross-sectional area [mm²] = 5mm x 5mm
L = 1000             # Total length of the bar [mm]
F = 200           # Applied load at the right end [N]
num_elements = 20   # Number of 1D elements

#===========================================
# Input Problem parameters
#===========================================
# E = float(input('Enter youngs modulus (MPa): '))
# A = float(input('Enter the area of cross-section (mm^2): '))
# L = float(input('Length of the Bar (mm): '))
# F = float(input('Force Applied (N): '))
# num_elements = int(input('Enter the number of elements to be generated: '))
num_nodes = num_elements + 1
element_length = L / num_elements
print(f'\nElement Length: {element_length} mm')

#===========================================
# Mesh and DOFs
#===========================================
K = np.zeros((num_nodes, num_nodes))  # Global stiffness matrix
f = np.zeros(num_nodes)               # Global force vector
u = np.zeros(num_nodes)               # Displacement vector

#===========================================
# Element stiffness matrix and assembly
#===========================================
k_local = (E * A / element_length) * np.array([[1, -1], [-1, 1]])

for i in range(num_elements):
    K[i:i+2, i:i+2] += k_local

#===========================================
# Apply Boundary Conditions and Loads
#===========================================
f[num_nodes-1] = F
# f[num_nodes-1] = 450          # Apply 100kN force at the right end
K_mod = K.copy()
f_mod = f.copy()

# Apply Dirichlet BC at node 0 (fixed end)
K_mod[0, :] = 0
K_mod[:, 0] = 0
K_mod[0, 0] = 1
f_mod[0] = 0

#===========================================
# Solve System
#===========================================
u = np.linalg.solve(K_mod, f_mod)
print('\nDisplacemnt (u):', u)

#===========================================
# Postprocessing
#===========================================
strain = np.zeros(num_elements)
stress = np.zeros(num_elements)
strain_energy_density = np.zeros(num_elements)

for i in range(num_elements):
    du = u[i+1] - u[i]
    strain[i] = du / element_length
    stress[i] = E * strain[i]
    strain_energy_density[i] = 0.5 * stress[i] * strain[i]

#===========================================
# Output
#===========================================
print("Nodal Displacements (in mm):")
print(np.round(u, 6))
print("\nElemental Stress (in MPa):")
print(np.round(stress, 3))
print("\nElemental Strain:")
print(np.round(strain, 6))
print("\nElemental Strain Energy Density (in J/mm³):")
print(np.round(strain_energy_density, 5))

#===========================================
# Plotting
#===========================================
x = np.linspace(0, L, num_elements)
print(x)


plt.figure(figsize=(5, 3))
plt.plot(x+element_length / 2, stress * 1, label="Stress (MPa)", marker='o')
plt.plot(x+element_length / 2, strain, label="Strain", marker='s')
plt.plot(x+element_length / 2, strain_energy_density, label="Strain Energy Density (J/mm³)", marker='^')
plt.xlabel("Length along bar (mm)")
plt.legend()
plt.title("1D Bar FEA: Stress, Strain & Strain Energy Density")
plt.grid(True)
# plt.show()

# --- Plot nodal displacement vs position ---
# Node positions
x = np.linspace(0, L, num_nodes)
plt.figure(figsize=(6, 3))
plt.plot(0, 0, marker='x', color='r', label='Analytical points')
plt.plot(L/2, 0.05578, marker='x', color='r', label='_nolegend_')
plt.plot(L, 0.111576, marker='x', color='r', label='_nolegend_')
plt.plot(x, u, color='green', linestyle='-', label='Nodal displacement')
plt.xlabel('Length along bar (mm)')
plt.ylabel('Displacement (mm)')
plt.title('Nodal Displacement (1D Bar)')
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()
