# Simlified code for plotting only SIR model

import numpy as np
import matplotlib.pyplot as plt
from odes.models import SIR
from odes.integrator import integrator


num_hosts = 5  # number of people in system
num_steps = 300
dt = 0.0001

time_scale = 1000.0  # can make time "move faster" by scaling these constants beyond [0, 1]
beta = time_scale*0.33  # infection rate
gamma = time_scale*0.2  # recovery rate
SIR_ODE = SIR(num_hosts, beta, gamma)
SIR_x0 = np.array([0.3, 0.5, 0.2])

SIR_stepper = integrator(SIR_ODE, SIR_x0, dt)
SIR_states = np.zeros((3, num_steps))
SIR_states[:, 0] = SIR_x0
time = np.zeros(num_steps)

for t in range(num_steps):
    print(f'Stepping timestep {t}/{num_steps}...')
    SIR_states[:, t] = SIR_stepper.step()
    time[t] = SIR_stepper.t

f, ax = plt.subplots()
ax.plot(time, SIR_states[0, :], label='S')
ax.plot(time, SIR_states[1, :], label='I')
ax.plot(time, SIR_states[2, :], label='R')
ax.legend()
ax.grid()
f.tight_layout()
plt.show()
