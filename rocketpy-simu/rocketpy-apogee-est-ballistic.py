import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import ode, simpson
from scipy.interpolate import interp1d
from rocketpy import Fluid, CylindricalTank, MassFlowRateBasedTank, HybridMotor
from rocketpy import Environment, Rocket, Flight

# ====================================================
# Part 1: Drag coefficient functions
# ====================================================
def Cd_function_ms():
    simulation = pd.read_excel('algorithms\\cd_simulation.xlsx')
    velocity_vals = simulation['m/s'].tolist()
    CD_vals = simulation['CD'].tolist()
    return interp1d(velocity_vals, CD_vals, kind='linear', fill_value="extrapolate")

def Cd_function_mach():
    simulation = pd.read_excel('algorithms\\cd_simulation.xlsx')
    velocity_vals = simulation['Mach'].tolist()
    CD_vals = simulation['CD'].tolist()
    return interp1d(velocity_vals, CD_vals, kind='linear', fill_value="extrapolate")

Cd_ms = Cd_function_ms()
Cd_mach = Cd_function_mach()


# ====================================================
# Part 2: Rocketpy simulation
# ====================================================

# --- Define Fluids and Tank ---
liquid_nox = Fluid(name="lNOx", density=786.6)
vapour_nox = Fluid(name="gNOx", density=159.4)

tank_radius = 150 / 2000  # in meters
tank_length = 0.7
tank_shape = CylindricalTank(tank_radius, tank_length)

burn_time = 7
nox_mass = 7.84
mass_flow = nox_mass / burn_time

oxidizer_tank = MassFlowRateBasedTank(
    name="oxidizer tank",
    geometry=tank_shape,
    flux_time=burn_time - 0.01,
    initial_liquid_mass=nox_mass,
    initial_gas_mass=0,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out= mass_flow,
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=0,
    liquid=liquid_nox,
    gas=vapour_nox,
)

# --- Define the Hybrid Motor ---
grain_length = 0.304
nozzle_length = 0.05185
plumbing_length = 0.4859   # including pre-cc and topcap/inj
post_cc_length = 0.0605
pre_cc_length = 0.039

fafnir = HybridMotor(
    thrust_source = 213 * 9.8 * mass_flow,  # using ISP=213
    dry_mass = 24.243,
    dry_inertia = (7.65, 7.65, 0.0845),
    nozzle_radius = 70 / 2000,
    grain_number = 1,
    grain_separation = 0,
    grain_outer_radius = 54.875 / 1000,
    grain_initial_inner_radius = 38.90 / 2000,
    grain_initial_height = grain_length,
    grain_density = 1.1,
    grains_center_of_mass_position = grain_length / 2 + nozzle_length + post_cc_length,
    center_of_dry_mass_position = 0.885,
    nozzle_position = 0,
    burn_time = burn_time,
    throat_radius = 15.875 / 2000,
)

fafnir.add_tank(
    tank=oxidizer_tank,
    position=post_cc_length + plumbing_length + grain_length + nozzle_length + tank_length / 2
)

# --- Define Environment ---
ground_level = 165
env = Environment(
    latitude=39.3897,
    longitude=-8.28896388889,
    elevation=ground_level,
    date=(2023, 10, 15, 12)
)
env.set_atmospheric_model("custom_atmosphere", wind_u=0, wind_v=-10)

# --- Define the Rocket ---
freya = Rocket(
    radius=0.077,
    mass=10.577,
    inertia=(13, 13, 0.0506),
    power_off_drag=Cd_mach, # Drag!
    power_on_drag=Cd_mach, # Drag!
    center_of_mass_without_motor=2.605,
    coordinate_system_orientation="tail_to_nose"
)

freya.add_motor(fafnir, 0.002)
freya.add_nose(length=0.26, kind="Von Karman", position=3.1556)
freya.add_trapezoidal_fins(
    4, root_chord=0.2, tip_chord=0.1, span=0.1, position=0.3, sweep_angle=25
)

spill_radius = 0.4 / 2
reefed_cd = 0.8
reefed_radius = 1.3 / 2
freya.add_parachute(
    'Reefed',
    cd_s=reefed_cd * math.pi * (reefed_radius**2 - spill_radius**2),
    trigger="apogee",
    lag=3
)

main_cd = 0.8
main_radius = 3.8 / 2
freya.add_parachute(
    'Main',
    cd_s=main_cd * math.pi * (main_radius**2 - spill_radius**2),
    trigger=200
)

# --- Run the Flight Simulation ---
test_flight = Flight(
    rocket=freya,
    environment=env,
    rail_length=12,
    inclination=84,
    heading=0,
    time_overshoot=False,
    terminate_on_apogee=True
)


times = test_flight.solution_array[:, 0]


# Find altitude at maximum velocity (end of burn phase) and apogee.
v_max_index = np.argmax(test_flight.solution_array[:, 6])
altitude_at_vmax = test_flight.solution_array[:, 3][v_max_index]

h_max = np.max(test_flight.solution_array[:, 3])
v_max = np.max(test_flight.solution_array[:, 6])


times = times[v_max_index+1:] # Only after burn phase


# ====================================================
# Part 3: Ballistic trajectory simulation functions
# ====================================================

def ode_ballistic(t, initial_state):
    # Get the time-dependent parameters using the absolute time t.

    y = initial_state[1]
    rho = env.density(t)
    m = freya.total_mass(t)
    g_val = 9.81
    Re = 6371.007181 * 10**3
    # Source: https://visibleearth.nasa.gov/images/104158/spain-and-portugal
    g = g_val * (Re/(Re + y))**2
    A = math.pi * freya.radius**2


    vx = initial_state[2]
    vy = initial_state[3]

    a = 3.43
    #Assume constant acceleration

    # Retrieve a drag coefficient based on the current velocity.
    v = np.sqrt(vx**2 + vy**2)
    Cd_val = float(Cd_ms(v)) if v > 15 else 0.4
    C_b = rho * v**2 / (2 * a)
    # C_b requires real-time acceleration data, using the value of 0.1sec earlier might give a rough estimate. 

    dxdt = vx
    dydt = vy
    dvxdt = -((rho * Cd_val * A) / (2 * m)) * vx * np.sqrt(vx**2 + vy**2)
    
    dvydt = -g - (rho * (dydt**2))/(2 * C_b)

    return [dxdt, dydt, dvxdt, dvydt]


def integrate_ballistic(initial_absolute_time, initial_state, duration=30, dt=0.1):
    solver = ode(lambda t, y: ode_ballistic(t, y))
    solver.set_integrator('dopri5')
    solver.set_initial_value(initial_state, initial_absolute_time)
    
    apogee_value = None
    # Run until the absolute time reaches initial_absolute_time + duration.
    while solver.successful() and solver.t < (initial_absolute_time + duration):
        # Check if vertical velocity becomes negative (i.e. we've passed apogee)
        if solver.t > initial_absolute_time and solver.y[3] < 0:
            # The second element of the state is the altitude
            apogee_value = solver.y[1]
            break
        solver.integrate(solver.t + dt)
    
    # If the loop ends without the velocity condition, return the last computed altitude.
    if apogee_value is None:
        apogee_value = solver.y[1]
    return apogee_value


# ====================================================
# Part 4: Call ballistic simulation for every time in Rocketpy simu
# ====================================================

apogee_predictions = []
apogee_predictions_errors = []

# Loop over each time t from the flight simulation
for t in times:
    state = test_flight.get_solution_at_time(t)
    altitude_ASL = state[3]            # z-coordinate (vertical position)
    h0 = altitude_ASL - env.elevation   # altitude above ground level
    v0 = np.linalg.norm(state[4:7])      # magnitude of velocity (vx, vy, vz)
    m_t = freya.total_mass(t)            # rocket mass at time t
    rho_t = env.density(altitude_ASL)     # local air density
    g_const = 9.81


    # Use state indices [x, y, vx, vy] for the ballistic simulation
    initial_state = [state[2], state[3], state[5], state[6]]

    # Integrate the ballistic trajectory from the current state
    apogee_value = integrate_ballistic(t, initial_state, duration=30, dt=0.1)


    apogee_predictions.append(apogee_value)
    apogee_predictions_errors.append(((apogee_value-h_max)/h_max)*100)

apogee_predictions = np.array(apogee_predictions)
apogee_predictions_errors = np.array(apogee_predictions_errors)


# ====================================================
# Part 5: Plot information
# ====================================================

print(f'Burn stops at: {altitude_at_vmax:.2f} m')
print(f'Maximum velocity: {v_max} m/s')
print(f'Apogee from simulation: {h_max:.2f} m')

# Plot: Apogee Prediction vs. Time
plt.figure()
plt.plot(times, apogee_predictions, 'r-', linewidth=2, label="Ballistic Apogee Prediction")
plt.xlim(0, 50)
plt.ylim(0, 5000)

plt.xlabel("Time (s)")
plt.ylabel("Predicted Apogee (m)")
plt.title("Apogee Prediction vs. Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Apogee Prediction Error vs. Time
plt.figure()
plt.plot(times, apogee_predictions_errors, 'r-', linewidth=2, label="Ballistic Apogee Prediction Error")
plt.xlim(0, 50)
plt.ylim(-20, 20)
plt.xlabel("Time (s)")
plt.ylabel("Predicted Apogee Error (%)")
plt.title("Apogee Prediction Error vs. Time")
plt.legend()
plt.grid(True)
plt.show()