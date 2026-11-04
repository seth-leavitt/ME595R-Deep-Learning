import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Number of data points
n = 10000

# Random seed for reproducibility
rng = np.random.default_rng(42)

# Generate extra samples so we can filter out transition flow
sample_count = int(n * 4.0)

# Fluid library (nominal properties near room temperature)
# Values chosen to provide meaningful Prandtl variation.
fluid_library = {
    "Water": {
        "rho": 997.0,
        "mu": 1.00e-3,
        "k": 0.60,
        "cp": 4180.0,
    },
    "Air": {
        "rho": 1.177,
        "mu": 1.85e-5,
        "k": 0.0263,
        "cp": 1007.0,
    },
    "Ethylene Glycol": {
        "rho": 1110.0,
        "mu": 1.61e-2,
        "k": 0.252,
        "cp": 2415.0,
    },
    "Light Oil": {
        "rho": 850.0,
        "mu": 5.00e-2,
        "k": 0.13,
        "cp": 2100.0,
    },
}

fluid_names = np.array(list(fluid_library.keys()), dtype=object)
fluid_choice_idx = rng.integers(0, len(fluid_names), sample_count)
fluid_name = fluid_names[fluid_choice_idx]

rho = np.array([fluid_library[name]["rho"] for name in fluid_name], dtype=float)
mu = np.array([fluid_library[name]["mu"] for name in fluid_name], dtype=float)
k = np.array([fluid_library[name]["k"] for name in fluid_name], dtype=float)
cp = np.array([fluid_library[name]["cp"] for name in fluid_name], dtype=float)
Pr = cp * mu / k

# Random geometry and operating conditions
D = rng.uniform(0.005, 0.10, sample_count)    # diameter (m)
L = rng.uniform(0.5, 20.0, sample_count)      # pipe length (m)
velocity = rng.uniform(0.01, 8.0, sample_count)  # m/s

# Bulk temperatures (K)
T_bulk = rng.uniform(290, 320, sample_count)
# Wall temperatures slightly higher (for heating case)
T_wall = T_bulk + rng.uniform(5, 300, sample_count)

# Calculate Reynolds number
Re = (rho * velocity * D) / mu

# Keep only clean laminar or turbulent cases (drop transition region)
is_laminar = Re < 2300
is_turbulent = Re > 4000
valid_flow = is_laminar | is_turbulent

# Thermal entry length criteria
# Laminar: L_th ~ 0.05 Re Pr D
# Turbulent: thermal development is rapid; use L_th ~ 10 D
L_th_critical = np.where(
    is_laminar,
    0.05 * Re * Pr * D,
    10.0 * D,
)
thermally_developed = L >= L_th_critical

# Nusselt number and correlation selection by regime + thermal development
Nu = np.empty(sample_count)
correlation_used = np.empty(sample_count, dtype=object)

# Laminar, thermally developed, constant wall temperature
mask_lam_fd = is_laminar & thermally_developed
Nu[mask_lam_fd] = 3.66
correlation_used[mask_lam_fd] = "Laminar fully developed (Nu=3.66)"

# Laminar, thermally developing, constant wall temperature approximation
mask_lam_dev = is_laminar & (~thermally_developed)
Nu[mask_lam_dev] = 1.86 * (
    (Re[mask_lam_dev] * Pr[mask_lam_dev] * D[mask_lam_dev] / L[mask_lam_dev]) ** (1.0 / 3.0)
)
correlation_used[mask_lam_dev] = "Laminar developing (Sieder-Tate entrance)"

# Turbulent, thermally developed: Dittus-Boelter heating
mask_turb_fd = is_turbulent & thermally_developed
Nu[mask_turb_fd] = 0.023 * (Re[mask_turb_fd] ** 0.8) * (Pr[mask_turb_fd] ** 0.4)
correlation_used[mask_turb_fd] = "Turbulent fully developed (Dittus-Boelter, heating)"

# Turbulent, thermally developing: Gnielinski + entrance correction
mask_turb_dev = is_turbulent & (~thermally_developed)
f_turb_dev = (0.79 * np.log(Re[mask_turb_dev]) - 1.64) ** -2
nu_gnielinski = (
    (f_turb_dev / 8.0) * (Re[mask_turb_dev] - 1000.0) * Pr[mask_turb_dev]
    / (1.0 + 12.7 * np.sqrt(f_turb_dev / 8.0) * ((Pr[mask_turb_dev] ** (2.0 / 3.0)) - 1.0))
)
entrance_correction = 1.0 + (D[mask_turb_dev] / L[mask_turb_dev]) ** (2.0 / 3.0)
Nu[mask_turb_dev] = nu_gnielinski * entrance_correction
correlation_used[mask_turb_dev] = "Turbulent developing (Gnielinski + entrance correction)"

# Remove transition-flow rows
Nu = Nu[valid_flow]
D = D[valid_flow]
L = L[valid_flow]
velocity = velocity[valid_flow]
T_bulk = T_bulk[valid_flow]
T_wall = T_wall[valid_flow]
Re = Re[valid_flow]
rho = rho[valid_flow]
mu = mu[valid_flow]
k = k[valid_flow]
cp = cp[valid_flow]
Pr = Pr[valid_flow]
fluid_name = fluid_name[valid_flow]
L_th_critical = L_th_critical[valid_flow]
thermally_developed = thermally_developed[valid_flow]
correlation_used = correlation_used[valid_flow]

# Truncate to requested dataset size
if len(Re) < n:
    raise ValueError("Not enough non-transition samples generated. Increase sample_count.")

Nu = Nu[:n]
D = D[:n]
L = L[:n]
velocity = velocity[:n]
T_bulk = T_bulk[:n]
T_wall = T_wall[:n]
Re = Re[:n]
rho = rho[:n]
mu = mu[:n]
k = k[:n]
cp = cp[:n]
Pr = Pr[:n]
fluid_name = fluid_name[:n]
L_th_critical = L_th_critical[:n]
thermally_developed = thermally_developed[:n]
correlation_used = correlation_used[:n]

# Convection coefficient
h = (Nu * k) / D

# Heat flux (q = h * delta_T)
q = h * (T_wall - T_bulk)

# KAN symbolic model prediction (from learned symbolic expression)
# x_1..x_5 correspond to [D, L/D, Re, Pr, k] after MinMax scaling to [0.05, 0.95].
KAN_SYMBOLIC_EXPR = (
    "0.52773*x_2 + 0.99011*x_5 + 0.0193*(-2.31368*x_3 + 0.40957*x_4 - 1.57963*x_5 - "
    "0.03582*log(9.8 - 9.9774*x_2) + 0.62764*log(2.2084*x_1 + 8.23744) - 1.55591)*"
    "(0.04273*x_4 + 0.49298*log(5.84728 - 1.20784*x_2) + 3.08278*log(8.96528 - 0.60808*x_1) - "
    "62.24269*log(0.58808*x_3 + 6.26064) - 27.60213*log(0.40816*x_5 + 9.54336) + 168.84143) - "
    "8.83037*log(5.50976 - 0.20792*x_4) - 0.71593*log(0.976*x_1 + 0.1968) + "
    "8.4216*log(0.62*x_3 + 8.18008) + 1.23403*log(-8.8076*(-0.00306*log(9.28824 - 9.32824*x_5) - "
    "0.00917*log(10.0*x_2 + 0.2) + 13.29853*log(4.22216*x_3 + 7.64464) - "
    "0.0244*log(9.85088*x_4 + 0.60392) - 27.20971)*(-1.61267*log(5.00488 - 1.53904*x_1) - "
    "8.86709*log(6.32488 - 2.4468*x_3) - 0.13437*log(7.71704 - 8.11312*x_5) - "
    "0.00897*log(4.42816*x_2 - 0.19984) - 24.28324*log(1.42856*x_4 + 9.5956) + 62.90492) - "
    "6.07048) + 0.2377*log(9.37432*(-1.11008*x_1 + 1.6295*x_3 + 0.35687*x_4 - "
    "13.19582*log(8.37464 - 0.204*x_5) - 0.12603*log(3.95896*x_2 + 2.77528) + 28.09479)*"
    "(1.18356*x_3 - 1.68443*x_4 - 0.2521*log(9.38232 - 3.59536*x_2) - "
    "0.04715*log(9.68808 - 7.06008*x_5) - 8.73184*log(0.18424*x_1 + 2.22376) + 7.35949) + "
    "8.57448) + 0.71325*log(-3.98432*(-0.24514*x_2 + 0.25017*x_3 + 1.87005*log(1.0044*x_1 + 8.81936) - "
    "7.10415*log(1.61568*x_4 + 9.48608) - 23.57219*log(0.41192*x_5 + 7.35872) + 58.72303)*"
    "(0.12416*x_2 - 1.22299*x_3 - 1.07818*x_5 + 3.32658*log(2.68064 - 0.57528*x_4) - "
    "2.97193*log(0.40384*x_1 + 7.38464) + 2.83489) + 9.7528) - 9.59155"
)

# This symbolic model was trained on log(h), then standardized.
KAN_SYMBOLIC_IS_LOG_TARGET = True

feature_matrix = np.column_stack([D, L / D, Re, Pr, k])
scaler_X = MinMaxScaler(feature_range=(0.05, 0.95))
X_scaled = scaler_X.fit_transform(feature_matrix)
x_1 = X_scaled[:, 0]
x_2 = X_scaled[:, 1]
x_3 = X_scaled[:, 2]
x_4 = X_scaled[:, 3]
x_5 = X_scaled[:, 4]

target_for_scaler = np.log(h).reshape(-1, 1) if KAN_SYMBOLIC_IS_LOG_TARGET else h.reshape(-1, 1)
scaler_y = StandardScaler()
scaler_y.fit(target_for_scaler)

with np.errstate(all="ignore"):
    y_symbolic_scaled = eval(
        KAN_SYMBOLIC_EXPR,
        {"__builtins__": {}, "log": np.log},
        {"x_1": x_1, "x_2": x_2, "x_3": x_3, "x_4": x_4, "x_5": x_5},
    )

y_symbolic_model = scaler_y.inverse_transform(np.asarray(y_symbolic_scaled).reshape(-1, 1)).ravel()
h_symbolic = np.exp(y_symbolic_model) if KAN_SYMBOLIC_IS_LOG_TARGET else y_symbolic_model
kan_symbolic_valid = np.isfinite(h_symbolic) & (h_symbolic > 0.0)

# Create DataFrame
data = pd.DataFrame({
    "Fluid": fluid_name,
    "Diameter D (m)": D,
    "Length L (m)": L,
    "Velocity (m/s)": velocity,
    "Density rho (kg/m^3)": rho,
    "Dynamic Viscosity mu (Pa*s)": mu,
    "Thermal Conductivity k (W/m*K)": k,
    "Specific Heat cp (J/kg*K)": cp,
    "Prandtl Number": Pr,
    "Bulk Temperature (K)": T_bulk,
    "Wall Temperature (K)": T_wall,
    "Reynolds Number": Re,
    "Thermal Entry Critical Length (m)": L_th_critical,
    "Thermally Developed": thermally_developed,
    "Nusselt Number": Nu,
    "Convection Coefficient h (W/m^2K)": h,
    "KAN Symbolic Convection Coefficient h (W/m^2K)": h_symbolic,
    "KAN Symbolic Valid": kan_symbolic_valid,
    "KAN Symbolic Correlation Used": np.where(kan_symbolic_valid, "KAN Symbolic (Unified)", "Invalid"),
    "Heat Flux q (W/m^2)": q,
    "Correlation Used": correlation_used,
})

# Save to CSV
data.to_csv("spoofed_convection_data.csv", index=False)

# Display first few rows
print(data)
