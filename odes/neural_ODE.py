import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint


class squared_parametrization(nn.Module):
    def forward(self, W):
        return torch.square(W)


class abs_parametrization(nn.Module):
    def forward(self, W):
        return torch.abs(W)


class nUIV_rhs(nn.Module):
    '''
    networked UIV model object
    holds an ODE defining a network of in-host models.
    Defined using pytorch tensors for interface with Neural ODE library.
    beta, delta, c, and p for each host are randomly initialized
    transmissions between hosts are randomly initialized. Could be constrained if we wanted to enforce more structure.
    args:
        N (int): number of hosts
    '''
    def __init__(self, N: int):
        super(nUIV_rhs, self).__init__()
        self.N = N
        self.parametrization = abs_parametrization()
        self.betas = nn.Parameter(torch.rand((self.N,)))
        self.deltas = nn.Parameter(torch.rand((self.N,)))
        self.cs = nn.Parameter(torch.rand((self.N,)))
        self.ps = nn.Parameter(torch.rand((self.N,)))
        self.ts = nn.Parameter(torch.rand((self.N, self.N)))

    def forward(self, t, state):
        rhs = torch.zeros_like(state)  # (U, I, V) RHS's for each node

        for n in range(self.N):
            U, I, V = state[3*n], state[3*n+1], state[3*n+2]  # re-naming host state for convenience
            beta = self.parametrization(self.betas[n])
            delta = self.parametrization(self.deltas[n])
            p = self.parametrization(self.ps[n])
            c = self.parametrization(self.cs[n])

            rhs[3*n] = - beta * U * V  # U dynamics
            rhs[3*n + 1] = beta * U * V - delta * I  # I dynamics
            rhs[3*n + 2] = p*I - c*V  # V dynamics

            # add transmission between neighbors (including self)
            for nbr in range(n):
                rhs[3*n + 2] += self.ts[nbr, n]*state[3*nbr+2]
            for nbr in range(n+1, self.N):
                rhs[3*n + 2] += self.ts[nbr, n]*state[3*nbr+2]

        return rhs


class soft_threshold(nn.Module):
    '''
    function for mapping from a single host's UIV state to their SIR state.
    Currently defined as a linear map plus a soft-thresholding operator.
    '''
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(3, 3)  # linear map
        self.slope = nn.ParameterList([torch.tensor(10.0)+torch.rand(1) for i in range(3)])
        self.threshold = nn.ParameterList([torch.rand(1) for i in range(3)])

    def forward(self, UIV_host):
        SIR_host = self.W(UIV_host)
        for i in range(3):
            SIR_host[i] = 1.0/(1.0 + torch.exp(-self.slope[i]*(SIR_host[i]-self.threshold[i])))
        return SIR_host


class nUIV_NODE(nn.Module):
    def __init__(self, num_hosts: int):
        super().__init__()
        self.num_hosts = torch.tensor(num_hosts)
        self.nUIV_x0 = nn.Parameter(torch.rand(3*self.num_hosts))  # initialize a random initial state
        self.nUIV_dynamics = nUIV_rhs(self.num_hosts)  # initialize a random nUIV
        self.nUIV_to_SIR = soft_threshold()

    def simulate(self, times):
        solution = odeint_adjoint(self.nUIV_dynamics, self.nUIV_x0.abs(), times)
        device = times.device
        SIR = torch.zeros(3, len(times)).to(device)
        for t in range(len(times)):
            for i in range(self.num_hosts):
                SIR[:, t] += self.nUIV_to_SIR(solution[t, 3*i:3*i+3])
        return SIR/self.num_hosts

    def get_params(self):
        params = dict()
        with torch.no_grad():
            params['beta'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.betas.values)
            params['delta'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.deltas.values)
            params['p'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.ps.values)
            params['c'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.cs.values)
            params['t'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.ts.values)
            params['x0'] = self.nUIV_dynamics.parametrization(self.nUIV_x0.values)
