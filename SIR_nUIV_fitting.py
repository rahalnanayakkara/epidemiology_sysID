from odes.models import SIR
from odes.integrator import integrator
from odes.neural_ODE import nUIV_NODE
import torch
# import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
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


def lp_norm_loss(y, yhat, p=2):
    return torch.pow(torch.norm(y-yhat, p=p), p)


# setting up SIR reference data
num_hosts = 100
num_steps = 300
dt = 0.0001
torch.manual_seed(666)

time_scale = 1000.0  # can make time "move faster" by scaling these constants beyond [0, 1]
beta = time_scale*0.33  # infection rate
gamma = time_scale*0.2  # recovery rate
SIR_ODE = SIR(num_hosts, beta, gamma)
SIR_x0 = np.array([0.3, 0.5, 0.2])


# generate data
SIR_stepper = integrator(SIR_ODE, SIR_x0, dt)
SIR_train_data, time_train_data = generate_SIR_data(SIR_stepper, num_steps)


# build model and fit it
method = 'euler'
step_size = dt/2.0
# build model and fit it
device = 'cpu'  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
nonlinearity = False
model = nUIV_NODE(num_hosts, method=method, step_size=step_size, nonlinearity=nonlinearity).to(device)
UIV_time_scale = 1.0

# INITIALIZE MODEL WITH REASONABLE PARAMETERS
with torch.no_grad():
    U0 = 4E8  # taken from paper
    I0 = 5  # taken from paper
    V0 = 50  # taken from paper

    # the initial states will be drawn uniformly at random in the interval
    #  [U0 * (1-percent_interval), U0*(1+percent_interval)].

    percent_interval = 0.01  # what % around the mean value are we willing to permit randomness about?
    U0_u = 1E8   # U0*(1+percent_interval)
    U0_l = 9E8  # U0*(1-percent_interval)
    I0_u = I0*(1+percent_interval)
    I0_l = I0*(1-percent_interval)
    V0_u = 0.0  # V0*(1+percent_interval)
    V0_l = 100  # V0*(1-percent_interval)

    # all initial states are in interval [0, 1].  Need to center them on U0, I0, V0 and scale them in the interval
    nUIV_x0_initial = torch.zeros_like(model.nUIV_x0)
    nUIV_x0_initial[::3] = U0_l + model.nUIV_x0[::3]*(U0_u - U0_l)
    nUIV_x0_initial[1::3] = I0_l + model.nUIV_x0[1::3]*(I0_u - I0_l)
    nUIV_x0_initial[2::3] = V0_l + model.nUIV_x0[2::3]*(V0_u - V0_l)
    model.nUIV_x0 = nUIV_x0_initial

    beta_l = UIV_time_scale*0.01E-8
    beta_u = UIV_time_scale*30E-8
    beta_m = UIV_time_scale*3E-8

    model.nUIV_dynamics.betas = beta_l + model.nUIV_dynamics.betas*(beta_u - beta_l)

    delta_l = UIV_time_scale*0.5
    delta_u = UIV_time_scale*3.0
    delta_m = UIV_time_scale*1.5

    model.nUIV_dynamics.deltas = delta_l + model.nUIV_dynamics.deltas*(delta_u - delta_l)

    p_l = UIV_time_scale*0.2
    p_u = UIV_time_scale*35.0  # 350
    p_m = UIV_time_scale*4.0

    model.nUIV_dynamics.ps = p_l + model.nUIV_dynamics.ps*(p_u-p_l)

    R0_l = UIV_time_scale*4.0
    R0_u = UIV_time_scale*70.0
    R0_m = UIV_time_scale*18.0

    c_l = U0*p_l*beta_l/(R0_l*delta_l)
    c_u = U0*p_u*beta_u/(R0_u*delta_u)
    c_m = U0*p_m*beta_m/(R0_m*delta_m)

    model.nUIV_dynamics.ts = c_l + model.nUIV_dynamics.ts*(c_u - c_l)
    if nonlinearity:
        # model.nUIV_to_SIR.threshold[0] = 100
        # model.nUIV_to_SIR.threshold[1] = 100
        # model.nUIV_to_SIR.threshold[2] = 100
        model.nUIV_to_SIR.threshold.data = torch.tensor(100.0)
    else:
        model.nUIV_to_SIR.W.weight.data[:, 0] = model.nUIV_to_SIR.W.weight.data[:, 0]*1E-9  # Need to drastically re-scale the UIV->SIR map
        model.nUIV_to_SIR.W.weight.data[:, 1] = model.nUIV_to_SIR.W.weight.data[:, 1]*1E-9
        model.nUIV_to_SIR.W.weight.data[:, 2] = model.nUIV_to_SIR.W.weight.data[:, 2]*1E-9


num_epochs = 200
lr = 1E-9
weight_decay = 0.0
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
only_I_fit = lambda y, yhat: torch.nn.functional.mse_loss(y[1, :], yhat[1, :])  # torch.nn.functional.l1_loss(y[1, :], yhat[1, :])
Lp_loss = lambda y, yhat: torch.nn.functional.mse_loss(y, yhat)  # , p=2)  # nn.L1Loss()

loss_function = Lp_loss

for epoch in range(num_epochs):
    optimizer.zero_grad()
    SIR_est = model.simulate(time_train_data.to(device)).to(device)
    loss = loss_function(SIR_est, SIR_train_data.to(device))
    loss_val = loss.item()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step(loss_val)

    print(f'Epoch {epoch}, loss value: {loss_val}.')
    if torch.isnan(loss):
        raise ValueError('Found NaN loss, exiting...')

# print(model.nUIV_x0)
# print(model.nUIV_dynamics.ts)

nUIV_params = model.get_params()
SIR_params = {'beta': beta,
              'gamma': gamma,
              'x0': SIR_x0,
              'num_hosts': num_hosts}

sim_params = {'SIR': SIR_params,
              'nUIV': nUIV_params}

path = './tmp/'
# path = '/content/tmp/'
if not os.path.exists(path):
    os.mkdir(path)
filename = os.path.join(path, 'params.p')
with open(filename, 'wb') as f:
    pickle.dump(sim_params, f)

np.set_printoptions(threshold=np.inf)
filename = os.path.join(path, 'params.txt')
with open(filename, 'w') as f:
    f.write('SIR MODEL PARAMETERS\n')
    for key, value in SIR_params.items():
        f.write(f'{key} : {value}\n')
    f.write('NUIV MODEL PARAMETERS\n')
    for key, value in nUIV_params.items():
        f.write(f'{key} : {value}\n')

# TODO: Write testing block to visualize the quality of the fit ODE
# First, reset the SIR model, change its time step
SIR_stepper.reset()
total_time = dt*num_steps
# SIR_stepper.dt = 0.001

# simulate for twice as far into future for testing
num_steps = int(2*total_time/SIR_stepper.dt)
SIR_test_data, time_test_data = generate_SIR_data(SIR_stepper, num_steps)

with torch.no_grad():
    SIR_train_data_est = model.simulate(time_train_data.to(device)).detach().cpu().numpy()
    SIR_test_data_est = model.simulate(time_test_data.to(device)).detach().cpu().numpy()

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
path = './tmp/'
# path = '/content/tmp/'
if not os.path.exists(path):
    os.mkdir(path)
filename = os.path.join(path, 'last_run_pytorch.png')
f.savefig(filename)
plt.show()
