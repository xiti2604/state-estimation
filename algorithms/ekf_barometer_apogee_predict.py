import numpy as np
import pandas as pd
from scipy.integrate import ode
from scipy.interpolate import interp1d

# Default definitions for Q, R, P
accelerometer_noise = 0.3
barometer_noise     = 1.0 # Should be lower than accelerometer, noise needs fixing.
P = np.eye(3) * 0.01
Q = np.eye(3) * (accelerometer_noise**2)
R = np.eye(1) * (barometer_noise**2)

constants = [9.81, 1.225, 20, 0.0186] # g, rho, m_dry, A

# Drag Coefficient
def Cd_function(): # assumption: angle of attack = 0
    simulation = pd.read_excel('algorithms\\cd_simulation.xlsx') 
    velocity_vals = simulation['m/s'].tolist()
    CD_vals = simulation['CD'].tolist()

    f_interp = interp1d(velocity_vals, CD_vals, kind='linear')  # 'linear' interpolation

    return f_interp

Cd_f = Cd_function()


# A matrix
def get_A(dt):
    return np.array([
        [1, dt, 0],
        [0, 1,  0],
        [0, 0,  0]
    ])

# B matrix
def get_B(dt):
    return np.array([
        [0,       0, 0.5*(dt**2)],
        [0,       0, dt],
        [0,       0, 1]
    ])

# Prediction step KF
def predict(x, P, Q, u, dt):
    g = 9.81
    az = -u[2] - g  # net acceleration (subtract gravity)
    A  = get_A(dt)
    B  = get_B(dt)
    x  = A @ x + B @ np.array([[az]])
    P  = A @ P @ A.T + Q
    return x, P

# h: Predicts pressure from altitude using the barometric formula.
#    This function serves as the nonlinear measurement function in the Kalman filter,
#    mapping the state (altitude) to a predicted pressure value.
def h(h_est):
    P0 = 101325.0
    T0 = 288.15
    L  = 0.0065
    g_const = 9.81
    R_const = 287.05
    base = 1 - (L * h_est / T0)
    if base < 1e-6:
        base = 1e-6
    return P0 * (base ** (g_const / (R_const * L)))

# H: Computes the Jacobian of the measurement function h with respect to altitude.
# Compute the Jacobian (measurement matrix) for the nonlinear measurement function.  
# The barometric measurement function that predicts pressure from altitude is given by: 
#     p(h) = P0 * (1 - (L * h) / T0)^(g / (R * L))
def H(h_est):
    P0 = 101325.0
    T0 = 288.15
    L  = 0.0065
    g_const = 9.81
    R_const = 287.05
    base = 1 - (L * h_est / T0)
    if base < 1e-6:
        base = 1e-6
    exponent = (g_const / (R_const * L)) - 1.0
    dpdh = -P0 * (g_const / (R_const * T0)) * (base ** exponent)
    return np.array([[dpdh, 0, 0]])

# correction step
def correct(x, P, R, pbaro):
    z = np.array([[pbaro]])  # raw pressure directly
    h_est = x[0,0]
    z_pred = h(h_est)
    H_jac = H(h_est)
    
    y = z - np.array([[z_pred]])
    S = H_jac @ P @ H_jac.T + R
    K = P @ H_jac.T @ np.linalg.inv(S)
    
    x = x + K @ y
    P = (np.eye(3) - K @ H_jac) @ P
    return x, P

# Apogee estimator
def ode_ballistic(t, state, constants):
    # [rho, m, g_val, A]
    g_ground = constants[0]
    rho = constants[1]
    m_dry = constants[2]
    A = constants[3]
    
    Re = 6371.007181 * 10**3
    # Source: https://visibleearth.nasa.gov/images/104158/spain-and-portugal
    g = g_ground * (Re/(Re + y))**2
    
    vy = state[2]
    vz = state[3]
    v = np.sqrt((vy**2)+(vz**2))

    # Retrieve a drag coefficient based on the current velocity.
    Cd_val = float(Cd_f(v)) if v > 15 else 0.4
    
    #Acceleration from drag. 
    a = (rho * Cd_val * A * v**2) / (2 * m)

    # Retrieve a ballistic coefficient based on the current velocity and acceleration.
    C_b = rho * v**2 / (2 * a)

    dydt = vy
    dzdt = vz
    dvydt = 0    #Unimportant
    dvzdt = -g - (rho * (dzdt**2))/(2 * C_b)
    
    return [dydt, dzdt, dvydt, dvzdt]


def integrate_ballistic(initial_absolute_time, initial_state, constants, duration=30, dt=0.1):
    solver = ode(lambda t, y: ode_ballistic(t, y, constants))
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



# Needs input of Q, R and initial P
def kf_runner(x, P, Q, R, u, pbaro, t, dt):
    x, P = predict(x, P, Q, u, dt)
    x, P = correct(x, P, R, pbaro)
    
    current_height = x[0].item()
    current_velocity = x[1].item()
    current_acceleration = x[2].item()

    initial_state = [0, current_height, 0, current_velocity] # y, z, vy, vz

    apogee = integrate_ballistic(t, initial_state, duration=30, dt=0.1)
    return x, P
