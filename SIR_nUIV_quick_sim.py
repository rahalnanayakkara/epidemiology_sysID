import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from odes.models import SIR, nUIV
from odes.integrator import integrator


num_hosts = 5  # number of people in system
edge_prob = 0.75  # probability of an edge between two
G = nx.fast_gnp_random_graph(num_hosts, edge_prob, directed=True)

# build some random exhalation coefficients
for n in G:
    exhale = np.random.rand()
    G.nodes[n]['beta'] = np.random.rand()
    G.nodes[n]['delta'] = np.random.rand()
    G.nodes[n]['p'] = np.random.rand()
    G.nodes[n]['c'] = np.random.rand()
    G.nodes[n]['state'] = np.random.rand(3)

    G.add_edge(n, n, transmit=-exhale)
    for nbr in G.successors(n):
        to_nbr = np.random.rand()*exhale
        exhale -= to_nbr
        G[n][nbr]['transmit'] = to_nbr

nUIV_ODE = nUIV(G)

num_steps = 300
dt = 0.0001

nUIV_x0 = nUIV_ODE.get_graph_state()
nUIV_stepper = integrator(nUIV_ODE, nUIV_x0, dt)
nUIV_states = np.zeros((3*num_hosts, num_steps))
nUIV_states[:, 0] = nUIV_x0

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
    # print(f'Stepping timestep {t}/{num_steps}...')
    SIR_states[:, t] = SIR_stepper.step()
    nUIV_states[:, t] = nUIV_stepper.step()
    time[t] = SIR_stepper.t

f, ax = plt.subplots()
ax.plot(time, SIR_states[0, :], label='S')
ax.plot(time, SIR_states[1, :], label='I')
ax.plot(time, SIR_states[2, :], label='R')
ax.legend()
ax.grid()
f.tight_layout()

host_num = 0
f2, ax2 = plt.subplots()
C = nUIV_states[3*host_num, 0] + nUIV_states[3*host_num, 1]
ax2.plot(time, nUIV_states[3*host_num, :]/C, label='U')
ax2.plot(time, nUIV_states[3*host_num+1, :]/C, label='I')
ax2.plot(time, nUIV_states[3*host_num+2, :], label='V')
ax2.legend()
ax2.grid()
f2.tight_layout()
plt.show()
