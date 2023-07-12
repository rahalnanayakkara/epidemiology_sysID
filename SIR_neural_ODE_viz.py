import pickle
import numpy as np
from odes.models import SIR, nUIV
from odes.integrator import integrator
import matplotlib.pyplot as plt
import networkx as nx


class soft_threshold:
    def __init__(self, **kwargs):
        self.slope = kwargs.pop('slope', 10.0)
        self.threshold = kwargs.pop('threshold', 0.0)
        print(f'initialized st w slope {self.slope} thresh {self.threshold}')

    def __call__(self, x):
        return 1.0/(1.0 + np.exp(-self.slope*(x-self.threshold)))


class nUIV_to_SIR:
    def __init__(self, params, num_hosts):
        self.num_hosts = num_hosts
        self.W = params['weight']
        self.b = params['bias']
        # load up the soft thresholds
        self.thresholds = []
        for i in range(3):
            self.thresholds.append(soft_threshold(**params['threshold_'+str(i)]))

    def __call__(self, nUIV):
        SIR = np.zeros(3,)
        for i in range(self.num_hosts):
            z = np.zeros(3,)
            for j in range(3):
                z[j] = self.thresholds[j](nUIV[3*i:3*i+3][j])
            SIR += self.W @ z + self.b
        return SIR/self.num_hosts


save_file = './tmp/params.p'
with open(save_file, 'rb') as f:
    params = pickle.load(f)

dt = 0.001
num_steps = 5000
time = np.zeros(num_steps,)

# LOADING SIR MODEL PARAMETERS
SIR_params = params['SIR']
num_hosts = SIR_params['num_hosts']
SIR_ODE = SIR(num_hosts, SIR_params['beta'], SIR_params['gamma'])
SIR_x0 = SIR_params['x0']
SIR_stepper = integrator(SIR_ODE, SIR_x0, dt)
SIR_states = np.zeros((3, num_steps))
SIR_states[:, 0] = SIR_x0

# LOADING nUIV PARAMETERS
nUIV_params = params['nUIV']
nUIV_x0 = nUIV_params['x0']
G = nx.DiGraph()
for i in range(num_hosts):
    attrs = {'state': nUIV_x0[3*i:3*i+3],
             'beta': nUIV_params['beta'][i],
             'delta': nUIV_params['delta'][i],
             'p': nUIV_params['p'][i],
             'c': nUIV_params['c'][i]}
    G.add_nodes_from([(i, attrs)])
for i in G.nodes():
    for j in G.nodes():
        G.add_edge(i, j, transmit=nUIV_params['t'][i, j])


nUIV_ODE = nUIV(G)
nUIV_stepper = integrator(nUIV_ODE, nUIV_x0, dt)
to_SIR = nUIV_to_SIR(nUIV_params['nUIV_to_SIR'], num_hosts)
nUIV_states = np.zeros((3*num_hosts, num_steps))
est_SIR_states = np.zeros((3, num_steps))
nUIV_states[:, 0] = nUIV_x0
est_SIR_states[:, 0] = to_SIR(nUIV_x0)

for t in range(num_steps):
    print(f'Stepping timestep {t+1}/{num_steps}...')
    SIR_states[:, t] = SIR_stepper.step()
    nUIV_states[:, t] = nUIV_stepper.step()
    est_SIR_states[:, t] = to_SIR(nUIV_states[:, t])
    time[t] = SIR_stepper.t

f, ax = plt.subplots()
ax.plot(time, SIR_states[0, :], label='S', color='red')
ax.plot(time, est_SIR_states[0, :], label='S est', linestyle='dashed', color='red')
ax.plot(time, SIR_states[1, :], label='I', color='green')
ax.plot(time, est_SIR_states[1, :], label='I est', linestyle='dashed', color='green')
ax.plot(time, SIR_states[2, :], label='R', color='blue')
ax.plot(time, est_SIR_states[2, :], label='R est', linestyle='dashed', color='blue')
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
