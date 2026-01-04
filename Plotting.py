import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# path to your .xlsx file
path = r"C:\Users\USER\Documents\Harmonic Loading.xlsx"

# read first sheet (use sheet_name="Sheet1" or index)
df = pd.read_excel(path, sheet_name="Final")
print(df)
Modal = df.iloc[12:1511,6]
Modal_Analysis = Modal.tolist()

Newmark = df.iloc[12:1511,206]
Newmark_Analysis = Newmark.tolist() 

Analytical = df.iloc[12:1511,7]
Analytical_Solution = Analytical.tolist()

time = df.iloc[12:1511,5]
Time = time.tolist()

t = np.array(Time, dtype=float)
y_anal = np.array(Analytical_Solution, dtype=float)
y_newm = np.array(Newmark_Analysis, dtype=float)
y_mod = np.array(Modal_Analysis, dtype=float)

# n = min(len(t), len(y_anal), len(y_newm), len(y_mod))
# t, y_anal, y_newm, y_mod = t[:n], y_anal[:n], y_newm[:n], y_mod[:n]

plt.figure(figsize=(10, 5))
plt.plot(t, y_newm, label='Newmark Method', color='blue', linewidth=1)
plt.plot(t, y_mod, label='Modal Analysis', color='red', linewidth=1)
plt.plot(t, y_anal, label='Analytical Solution', color='green', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Displacement vs Time for 1D Bar under Impulse Loading')
plt.legend(fontsize=12)
# plt.ylim(-5e-8, 5e-8)
plt.grid(True,linestyle='--')
plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\bar_displacement_Comparision_snapshot.jpeg', format='jpeg', dpi=1200)


plt.figure(figsize=(10, 5))
plt.plot(t, y_anal, label='Analytical Solution')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Displacement vs Time for 1D Bar under Impulse Loading')
plt.legend(fontsize=12)
# plt.ylim(-5e-8, 5e-8)
plt.grid(True)
plt.savefig(r'C:\Users\USER\OneDrive - IIT Hyderabad\Mtech Aerospace Engineering\Semester 3\Project Thesis\Results\bar_displacement_Analytical_snapshot.jpeg', format='jpeg', dpi=1200)

   
# inspect
print(df.shape)
print(df.head())