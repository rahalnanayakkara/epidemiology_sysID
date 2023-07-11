from odes.models import SIR
from odes.integrator import integrator
from odes.neural_ODE import nUIV_NODE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
num_steps = 10
dt = 0.05

beta = 0.1
gamma = 0.1
SIR_ODE = SIR(num_hosts, beta, gamma)
SIR_x0 = np.array([1.0, 1.0, 1.0])/3.


# generate data
SIR_stepper = integrator(SIR_ODE, SIR_x0, dt)
SIR_data, time_data = generate_SIR_data(SIR_stepper, num_steps)


# build model and fit it
model = nUIV_NODE(num_hosts)
num_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    SIR_est = model.simulate(time_data)
    loss = loss_function(SIR_est, SIR_data)
    loss_val = loss.item()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch}, loss value: {loss_val}.')

print(model.nUIV_x0)
print(model.nUIV_dynamics.ts)
