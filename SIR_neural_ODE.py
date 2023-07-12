from odes.models import SIR
from odes.integrator import integrator
from odes.neural_ODE import nUIV_NODE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle


# helper function to step the SIR model forward and generate a data set
def generate_SIR_data(model, num_steps):
    t = torch.zeros(num_steps)
    y = torch.zeros(3, num_steps)
    y[:, 0] = torch.from_numpy(model.x)
    t[0] = torch.tensor(0.0)
    for i in range(num_steps):
        y[:, i] = torch.from_numpy(model.step())
        t[i] = torch.tensor(model.t)
    return y, t


# setting up SIR reference data
num_hosts = 5
num_steps = 200
dt = 0.01
torch.manual_seed(0)

beta = 0.1
gamma = 0.99
SIR_ODE = SIR(num_hosts, beta, gamma)
SIR_x0 = np.array([1.0, 1.0, 1.0])/3.


# generate data
SIR_stepper = integrator(SIR_ODE, SIR_x0, dt)
SIR_train_data, time_train_data = generate_SIR_data(SIR_stepper, num_steps)


# build model and fit it
device = 'cpu'  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = nUIV_NODE(num_hosts, method='rk4').to(device)
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.0)
loss_function = nn.L1Loss()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    SIR_est = model.simulate(time_train_data.to(device))
    loss = loss_function(SIR_est, SIR_train_data.to(device))
    loss_val = loss.item()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch}, loss value: {loss_val}.')
    if loss_val == 'nan':
        raise ValueError('Found NaN loss, exiting...')

print(model.nUIV_x0)
print(model.nUIV_dynamics.ts)

nUIV_params = model.get_params()
SIR_params = {'beta': beta,
              'gamma': gamma,
              'x0': SIR_x0,
              'num_hosts': num_hosts}

sim_params = {'SIR': SIR_params,
              'nUIV': nUIV_params}

filename = './tmp/params.p'
with open(filename, 'wb') as f:
    pickle.dump(sim_params, f)

# TODO: Write testing block to visualize the quality of the fit ODE
# First, reset the SIR model, change its time step
SIR_stepper.reset()
SIR_stepper.dt = 0.001
num_steps = 5000
SIR_test_data, time_test_data = generate_SIR_data(SIR_stepper, num_steps)

with torch.no_grad():
    SIR_train_data_est = model.simulate(time_train_data.to(device)).detach()
    SIR_test_data_est = model.simulate(time_test_data.to(device)).detach()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
names = ['S', 'I', 'R']
colors = ['red', 'green', 'blue']
for i, name in enumerate(names):
    ax1.plot(time_train_data, SIR_train_data[i, :], lw=2, color=colors[i], label=name)
    ax1.plot(time_train_data, SIR_train_data_est[i, :], lw=2, color=colors[i], label=name+' est', linestyle='dashed')
    ax2.plot(time_test_data, SIR_test_data[i, :], lw=2, color=colors[i], label=name)
    ax2.plot(time_test_data, SIR_test_data_est[i, :], lw=2, color=colors[i], label=name+' est', linestyle='dashed')
ax1.set_title('Training')
ax2.set_title('Testing')
ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend()
f.tight_layout()
f.savefig('./tmp/last_run_pytorch.png')
plt.show()
