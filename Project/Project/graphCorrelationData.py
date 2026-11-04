import matplotlib
matplotlib.use('TkAgg')  # Better backend for interactive plots on macOS
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.lines import Line2D
from typing import Any


# Load the data from the CSV file
data = pandas.read_csv("spoofed_convection_data.csv")

# Extract the relevant columns
velocity = data["Velocity (m/s)"].to_numpy(dtype=float)
critical_length = data["Thermal Entry Critical Length (m)"].to_numpy(dtype=float)
convection_coefficient = data["Convection Coefficient h (W/m^2K)"].to_numpy(dtype=float)
kan_symbolic_h = data["KAN Symbolic Convection Coefficient h (W/m^2K)"].to_numpy(dtype=float)
correlation_used = np.array(data["Correlation Used"].astype(str).tolist(), dtype=object)
kan_symbolic_valid = data["KAN Symbolic Valid"]
if kan_symbolic_valid.dtype != bool:
    kan_symbolic_valid = kan_symbolic_valid.astype(str).str.lower().isin(["true", "1", "yes"])
kan_symbolic_valid = kan_symbolic_valid.to_numpy(dtype=bool)

thermally_developed = data["Thermally Developed"]
if thermally_developed.dtype != bool:
    thermally_developed = thermally_developed.astype(str).str.lower().isin(["true", "1", "yes"])
thermally_developed = thermally_developed.to_numpy(dtype=bool)

# Create a 3D scatter plot
fig = plt.figure()
ax: Any = fig.add_subplot(111, projection='3d')

unique_corr = sorted(set(correlation_used.tolist()))
cmap = plt.get_cmap("tab10")
color_map = {corr: cmap(i % 10) for i, corr in enumerate(unique_corr)}

for corr in unique_corr:
    for developed, marker in [(True, 'o'), (False, 'x')]:
        mask = (correlation_used == corr) & (thermally_developed == developed)
        if np.any(mask):
            ax.scatter3D(
                velocity[mask],
                critical_length[mask],
                convection_coefficient[mask],
                c=[color_map[corr]],
                marker=marker,
                s=20,
                alpha=0.8,
            )

ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('Thermal Entry Critical Length (m)')
ax.set_zlabel('Convection Coefficient h (W/m^2K)')
ax.set_title('Flow Cases: Correlation by Color, Development by Marker')

correlation_legend = [
    Line2D([0], [0], marker='o', color='w', label=corr,
           markerfacecolor=color_map[corr], markersize=8)
    for corr in unique_corr
]
marker_legend = [
    Line2D([0], [0], marker='o', color='black', linestyle='None', label='Thermally Developed'),
    Line2D([0], [0], marker='x', color='black', linestyle='None', label='Thermally Undeveloped'),
    Line2D([0], [0], marker='^', color='black', linestyle='None', label='KAN Symbolic (Valid)'),
    Line2D([0], [0], marker='1', color='black', linestyle='None', label='KAN Symbolic (Invalid)'),
]

legend1 = ax.legend(
    handles=correlation_legend,
    title='Correlation Used',
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0)
)
ax.add_artist(legend1)
ax.legend(
    handles=marker_legend,
    title='Thermal Region',
    loc='upper left',
    bbox_to_anchor=(1.02, 0.55)
)

# Overlay KAN symbolic predictions with dedicated markers and fixed color
for valid, marker in [(True, '^'), (False, '1')]:
    mask = kan_symbolic_valid == valid
    if np.any(mask):
        ax.scatter3D(
            velocity[mask],
            critical_length[mask],
            kan_symbolic_h[mask],
            c=['black'],
            marker=marker,
            s=24,
            alpha=0.9,
        )

plt.tight_layout()

# Enable interactive mode for mouse controls
plt.show()
