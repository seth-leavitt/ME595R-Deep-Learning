import numpy as np
import matplotlib.pyplot as plt


class Pancake:
    def __init__(
        self,
        temp=373.15,          # K, ~100 C fresh pancake
        radius=0.10,          # m
        thickness=0.01,       # m
        density=2700.0,       # kg/m^3
        cp=900.0              # J/(kg*K)
    ):
        self.temp = temp
        self.radius = radius
        self.thickness = thickness
        self.density = density
        self.cp = cp

    @property
    def area_face(self):
        return np.pi * self.radius**2

    @property
    def area_side(self):
        return 2 * np.pi * self.radius * self.thickness

    @property
    def volume(self):
        return self.area_face * self.thickness

    @property
    def mass(self):
        return self.density * self.volume

    @property
    def heat_capacity(self):
        return self.mass * self.cp


class PancakeStack:
    def __init__(
        self,
        air_temp=300.0,           # K
        plate_temp=300.0,         # K
        conductivity=0.5,         # W/(m*K), rougher pancake-like value
        h_top=12.0,               # W/(m^2*K)
        h_bottom=8.0,             # W/(m^2*K)
        h_side=10.0               # W/(m^2*K)
    ):
        self.pancakes = []        # index 0 = bottom pancake, index -1 = top pancake
        self.air_temp = air_temp
        self.plate_temp = plate_temp
        self.k = conductivity
        self.h_top = h_top
        self.h_bottom = h_bottom
        self.h_side = h_side

    def add_pancake(self, pancake):
        # add fresh pancake to the TOP of the stack
        self.pancakes.append(pancake)

    def contact_resistance(self, pancake):
        # center-to-center conduction approximation through one pancake thickness
        # R = L / (k A)
        return pancake.thickness / (self.k * pancake.area_face)

    def time_step(self, dt):
        if not self.pancakes:
            return

        n = len(self.pancakes)
        dTdt = np.zeros(n)

        for i, p in enumerate(self.pancakes):
            q_net = 0.0

            # heat exchange with pancake below
            if i > 0:
                p_below = self.pancakes[i - 1]
                R = self.contact_resistance(p)
                q_net += (p_below.temp - p.temp) / R

            # heat exchange with pancake above
            if i < n - 1:
                p_above = self.pancakes[i + 1]
                R = self.contact_resistance(p)
                q_net += (p_above.temp - p.temp) / R

            # bottom pancake loses heat to plate / room
            if i == 0:
                q_net += (self.plate_temp - p.temp)

            # top pancake loses heat from top surface to air
            if i == n - 1:
                q_net += self.h_top * p.area_face * (self.air_temp - p.temp)

            # every pancake loses heat from its side surface
            q_net += self.h_side * p.area_side * (self.air_temp - p.temp)

            dTdt[i] = q_net / p.heat_capacity

        # update all temperatures simultaneously
        for i in range(n):
            self.pancakes[i].temp += dTdt[i] * dt

    def temperatures_C(self):
        return [p.temp - 273.15 for p in self.pancakes]


# -----------------------------
# simulation
# -----------------------------
stack = PancakeStack(
    air_temp=300.0,        # room temp K (~26.85 C)
    plate_temp=300.0,
    conductivity=0.5,      # pancake is much less conductive than aluminum
    h_top=12.0,
    h_side=10.0
)

pancakeParamterCases = [
    {"temp": 373.15, "radius": 0.10, "thickness": 0.01, "density": 2700.0, "cp": 900.0, "time_per_pancake": 30.0, "label": "Case 1", "color": "blue"},
    {"temp": 373.15, "radius": 0.12, "thickness": 0.015, "density": 2500.0, "cp": 850.0, "time_per_pancake": 30.0, "label": "Case 2", "color": "orange"},
    {"temp": 373.15, "radius": 0.08, "thickness": 0.008, "density": 2900.0, "cp": 950.0, "time_per_pancake": 30.0, "label": "Case 3", "color": "green"},
    {"temp": 373.15, "radius": 0.11, "thickness": 0.012, "density": 2600.0, "cp": 920.0, "time_per_pancake": 30.0, "label": "Case 4", "color": "red"},
    {"temp": 373.15, "radius": 0.09, "thickness": 0.009, "density": 2800.0, "cp": 880.0, "time_per_pancake": 30.0, "label": "Case 5", "color": "purple"},
]

caseObservations = []

for params in pancakeParamterCases:
    stack.pancakes = [] # empty out the stack
    t = 0.0
    # start with one pancake
    stack.add_pancake(Pancake(params["temp"], params["radius"], params["thickness"], params["density"], params["cp"]))
    dt = 0.5  # seconds
    total_time = 30 * 60  # simulate 20 minutes
    time_per_pancake = params["time_per_pancake"]  # new pancake every 30 s
    next_add_time = time_per_pancake

    times = []
    bottom_temps = []
    top_temps = []
    stack_sizes = []

    while t <= total_time:
        # record
        times.append(t)
        bottom_temps.append(stack.pancakes[0].temp - 273.15)
        top_temps.append(stack.pancakes[-1].temp - 273.15)
        stack_sizes.append(len(stack.pancakes))

        # add new pancake every 30 seconds
        if t >= next_add_time:
            stack.add_pancake(Pancake(temp=373.15, radius=params["radius"],
                                      thickness=params["thickness"], density=params["density"],
                                      cp=params["cp"]))
            next_add_time += time_per_pancake

        # advance
        stack.time_step(dt)
        t += dt

    # print final pancake temperatures
    print("Final pancake temperatures (bottom -> top), in C:")
    print([round(T, 2) for T in stack.temperatures_C()])

    caseObservations.append({"label": params["label"], "color" : params["color"], "final_temps_C": stack.temperatures_C(), "stack_size": len(stack.pancakes),
                             "times": times, "bottom_temps": bottom_temps, "top_temps": top_temps, "stack_sizes": stack_sizes})

# plot
plt.figure(figsize=(10, 6))
legend_entries = []
for case in caseObservations:
    l1 =plt.plot(case["times"], case["bottom_temps"], label=f"{case['label']}", color=case["color"])
    plt.plot(case["times"], case["top_temps"], label=f"{case['label']}", color=case["color"], linestyle="--")
    legend_entries.append(l1)
plt.xlabel("Time (s)")
plt.ylabel("Temperature (C)")
plt.title("Temperature of a stack of pancakes over time (1 pancake added every 30 s)")

t_count = caseObservations[0]["times"]
l2 = plt.plot(t_count, np.ones_like(t_count)*55, label=f"Threshold for a \"cold\" pancake (55 C)", linestyle="--", color="gray")
from matplotlib.lines import Line2D

# --- Legend 1: colors = cases ---
color_handles = [Line2D([0], [0], color=case["color"]) for case in caseObservations]

legend1 = plt.legend(
    color_handles,
    [case["label"] for case in caseObservations],
    loc="upper right"
)

plt.gca().add_artist(legend1)


# --- Legend 2: linestyles = meaning ---
style_handles = [
    Line2D([0], [0], color='black', linestyle='-'),
    Line2D([0], [0], color='black', linestyle='--'),
    Line2D([0], [0], color='gray', linestyle='--')
]

legend2 = plt.legend(
    style_handles,
    ["Bottom pancake", "Top pancake", "Cold threshold"],
    loc="upper right",
    bbox_to_anchor=(1, 0.75)
)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(times, stack_sizes)
plt.xlabel("Time (s)")
plt.ylabel("Number of pancakes")
plt.title("Stack size over time")
plt.grid(True)
plt.show()