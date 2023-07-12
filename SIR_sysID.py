import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from odes.models import SIR, nUIV
from odes.integrator import integrator
import torch
import torch.nn as nn
from functorch import jacfwd


class nUIV_to_SIR(nn.Module):
    '''
    Callable class that converts a networked UIV model state to an SIR state
    '''
    def __init__(self, num_hosts, **kwargs):
        self.num_hosts = num_hosts

        self.slope = kwargs.pop('slope', 10.0)
        self.threshold = kwargs.pop('threshold', 1e3)

    def forward(self, UIV_state):
        SIR_state = torch.zeros(3,)
        '''
        W = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],conda
                    [0.0, 0.0, 1.0]])
        '''
        for n in range(num_hosts):
            SIR_state[2] += 1.0
            SIR_state[0] += self.soft_threshold(UIV_state[3*n], self.slope, self.threshold)
            SIR_state[2] -= self.soft_threshold(UIV_state[3*n], self.slope, self.threshold)
            SIR_state[1] += self.soft_threshold(UIV_state[3*n+1], self.slope, self.threshold)
            SIR_state[2] -= self.soft_threshold(UIV_state[3*n+1], self.slope, self.threshold)
        return SIR_state/self.num_hosts

    def soft_threshold(self, x, slope=10.0, threshold=0.0):
        return 1.0/(1.0 + torch.exp(slope*x - threshold))

    def jacobian(self, UIV_state):
        return jacfwd(self.forward, argnums=1)(UIV_state).numpy()


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

num_steps = 200
dt = 0.01

nUIV_x0 = nUIV_ODE.get_graph_state()
nUIV_stepper = integrator(nUIV_ODE, nUIV_x0, dt)
nUIV_states = np.zeros((3*num_hosts, num_steps))
nUIV_states[:, 0] = nUIV_x0

beta = 0.01
gamma = 0.999
SIR_ODE = SIR(num_hosts, beta, gamma)
SIR_x0 = np.array([1.0, 1.0, 1.0])/3.


SIR_stepper = integrator(SIR_ODE, SIR_x0, dt)
SIR_states = np.zeros((3, num_steps))
SIR_states[:, 0] = SIR_x0
time = np.zeros(num_steps)

for t in range(num_steps):
    print(f'Stepping timestep {t}/{num_steps}...')
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
