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
    return torch.norm(y-yhat, p=p)


# setting up SIR reference data
num_hosts = 50
num_steps = 1000
dt = 0.01
torch.manual_seed(666)

time_scale = 75.0  # can make time "move faster" by scaling these constants beyond [0, 1]
beta = time_scale*0.9  # infection rate
gamma = time_scale*0.01  # recovery rate
SIR_ODE = SIR(num_hosts, beta, gamma)
SIR_x0 = np.array([0.99, 0.01, 0.0])


# generate data
SIR_stepper = integrator(SIR_ODE, SIR_x0, dt)
SIR_train_data, time_train_data = generate_SIR_data(SIR_stepper, num_steps)


# build model and fit it
method = 'euler'
step_size = 2*dt
# build model and fit it
device = 'cpu'  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = nUIV_NODE(num_hosts, method=method, step_size=step_size).to(device)
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=1e-1, weight_decay=0.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
loss_function = lambda y, yhat: lp_norm_loss(y, yhat, p=2)  # nn.L1Loss()

#UIV_x0_train = torch.zeros(3*num_hosts)
#UIV_x0_train[::3] = 10**9
#UIV_x0_train[2::3] = 10
#UIV_x0_train = UIV_x0_train.T
#UIV_U0_train = UIV_x0_train[::3]
#UIV_I0_train = UIV_x0_train[1::3]
#UIV_V0_train = UIV_x0_train[2::3]
#com_train = torch.cat((SIR_train_data[1,:],1/5*torch.log10(UIV_U0_train),1/5*UIV_I0_train,1/5*torch.log10(UIV_V0_train)),dim=0)

train_beta = torch.tensor([1.35e-7,1.26e-7,5.24e-7,7.92e-10,1.51e-7,5.74e-10,1.23e-7,2.62e-9,3.08e-10])
log_train_beta = torch.log10(train_beta)
train_delta = torch.tensor([0.61,0.81,0.51,1.21,2.01,0.81,0.91,1.61,2.01])
#train_p = torch.tensor([0.2,0.2,0.2,361.6,0.2,382,0.2,278.2,299])
#log_train_p = torch.log10(train_p)
train_p = torch.tensor([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
train_c = 2.4*torch.ones(9)

#train_params = torch.tensor([torch.mean(log_train_beta),torch.mean(train_delta),torch.mean(log_train_p),torch.mean(train_c)])
train_params = torch.cat((torch.mean(log_train_beta)*torch.ones(num_hosts),torch.mean(train_delta)*torch.ones(num_hosts),torch.mean(train_p)*torch.ones(num_hosts),torch.mean(train_c)*torch.ones(num_hosts)))

com_train = torch.cat((SIR_train_data[1,:],0.01*train_params))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    SIR_est = model.simulate(time_train_data.to(device)).to(device)
    #UIV_x0_est = model.nUIV_x0
    #UIV_x0_est = UIV_x0_est.T
    #UIV_U0_est = UIV_x0_est[::3]
    #UIV_I0_est = UIV_x0_est[1::3]
    #UIV_V0_est = UIV_x0_est[2::3]
    #com_est = torch.cat((SIR_est[1,:],1/5*torch.log10(UIV_U0_est),1/5*UIV_I0_est,1/5*torch.log10(UIV_V0_est)),dim=0)

    beta_est = model.nUIV_dynamics.betas
    log_beta_est = torch.log10(beta_est)
    delta_est = model.nUIV_dynamics.deltas
    p_est = model.nUIV_dynamics.ps
    log_p_est = torch.log10(p_est)
    c_est = model.nUIV_dynamics.cs

    #params_est = torch.tensor([torch.mean(log_beta_est),torch.mean(delta_est),torch.mean(log_p_est),torch.mean(c_est)])
    params_est = torch.cat((log_beta_est,delta_est,p_est,c_est))

    com_est = torch.cat((SIR_est[1,:],0.01*params_est))

    loss = loss_function(com_est, com_train.to(device))
    #loss = loss_function(SIR_est, SIR_train_data.to(device))
    loss_val = loss.item()
    loss.backward()
    optimizer.step()
    scheduler.step(loss_val)

    print(f'Epoch {epoch}, loss value: {loss_val}.')
    if torch.isnan(loss):
        raise ValueError('Found NaN loss, exiting...')

# print(model.nUIV_x0)
# print(model.nUIV_dynamics.ts)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, Gradient: {param.grad}")


nUIV_params = model.get_params()
SIR_params = {'beta': beta,
              'gamma': gamma,
              'x0': SIR_x0,
              'num_hosts': num_hosts}

sim_params = {'SIR': SIR_params,
              'nUIV': nUIV_params}

print(nUIV_params)

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
    # f.write('SIMULATION PARAMATERS:\n')
    # for key, value in sim_params.items():
    #     f.write(f'{key} : {value}\n')
    f.write('SIR MODEL PARAMETERS\n')
    for key, value in SIR_params.items():
        f.write(f'{key} : {value}\n')
    f.write('NUIV MODEL PARAMETERS\n')
    for key, value in nUIV_params.items():
        f.write(f'{key} : {value}\n')

# TODO: Write testing block to visualize the quality of the fit ODE
# First, reset the SIR model, change its time step
SIR_stepper.reset()
SIR_stepper.dt = 0.001
num_steps = 20000
SIR_test_data, time_test_data = generate_SIR_data(SIR_stepper, num_steps)

with torch.no_grad():
    SIR_train_data_est = model.simulate(time_train_data.to(device)).detach().cpu().numpy()
    SIR_test_data_est = model.simulate(time_test_data.to(device)).detach().cpu().numpy()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
names = ['S', 'I', 'R']
colors = ['red', 'green', 'blue']
'''
for i, name in enumerate(names):
    ax1.plot(time_train_data, SIR_train_data[i, :], lw=2, color=colors[i], label=name)
    ax1.plot(time_train_data, SIR_train_data_est[i, :], lw=2, color=colors[i], label=name+' est', linestyle='dashed')
    ax2.plot(time_test_data, SIR_test_data[i, :], lw=2, color=colors[i], label=name)
    ax2.plot(time_test_data, SIR_test_data_est[i, :], lw=2, color=colors[i], label=name+' est', linestyle='dashed')
'''
ax1.plot(time_train_data, SIR_train_data[1, :], lw=2, color=colors[1], label=names[1])
ax1.plot(time_train_data, SIR_train_data_est[1, :], lw=2, color=colors[1], label=names[1]+' est', linestyle='dashed')
ax2.plot(time_test_data, SIR_test_data[1, :], lw=2, color=colors[1], label=names[1])
ax2.plot(time_test_data, SIR_test_data_est[1, :], lw=2, color=colors[1], label=names[1]+' est', linestyle='dashed')
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
