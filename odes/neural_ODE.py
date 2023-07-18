import torch
import torch.nn as nn
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint


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
        self.parametrization = squared_parametrization()
        self.betas = nn.Parameter(torch.rand((self.N,)))
        self.deltas = nn.Parameter(torch.rand((self.N,)))
        self.cs = nn.Parameter(torch.rand((self.N,)))
        self.ps = nn.Parameter(torch.rand((self.N,)))
        self.ts = nn.Parameter(torch.rand((self.N, self.N))*0.5)
        # elf.normalization = torch.ones((self.N,))  # for projecting onto constraints

    def forward(self, t, state):
        rhs = torch.zeros_like(state)  # (U, I, V) RHS's for each node
        normalization = self.compute_normalization()

        for n in range(self.N):
            U, I, V = state[3*n], state[3*n+1], state[3*n+2]  # re-naming host state for convenience
            beta = self.parametrization(self.betas[n])
            delta = self.parametrization(self.deltas[n])
            p = self.parametrization(self.ps[n])
            c = self.parametrization(self.cs[n])

            rhs[3*n] = - beta * U * V  # U dynamics
            rhs[3*n + 1] = beta * U * V - delta * I  # I dynamics
            rhs[3*n + 2] = p*I - c*V - self.parametrization(self.ts[n, n])  # V dynamics

            # add transmission between neighbors (including self)
            # normalize so that total intake is bounded by
            for nbr in range(n):
                rhs[3*n + 2] += normalization[nbr]*self.parametrization(self.ts[nbr, n])*state[3*nbr+2]
            for nbr in range(n+1, self.N):
                rhs[3*n + 2] += normalization[nbr]*self.parametrization(self.ts[nbr, n])*state[3*nbr+2]

        return rhs

    def compute_normalization(self):
        normalization = torch.zeros_like(self.betas)
        for n in range(self.N):
            normalization[n] = torch.min(self.parametrization(self.ts[n, n]) /
                                         (torch.sum(self.parametrization(self.ts[n, :]))
                                          - self.parametrization(self.ts[n, n])), torch.tensor(1.0))
        return normalization

    def normalize_ts(self):
        self.compute_normalization()
        for n in range(self.N):
            self.ts[n, :n] *= self.normalization[n]
            self.ts[n, n+1:] *= self.normalization[n]


class soft_threshold(nn.Module):
    '''
    function for mapping from a single host's UIV state to their SIR state.
    Currently defined as a linear map plus a soft-thresholding operator.
    '''
    def __init__(self, bias=False, nonlinearity=False):
        super().__init__()
        self.W = nn.Linear(3, 3, bias=bias)  # linear map
        if nonlinearity:
            self.slope = nn.ParameterList([torch.tensor(10.0)+torch.rand(1) for i in range(3)])
            self.threshold = nn.ParameterList([torch.rand(1) for i in range(3)])

    def forward(self, UIV_host):
        # SIR_host = UIV_host  # self.W(UIV_host)
        # SIR_host = torch.zeros(3, device=UIV_host.device)
        # for i in range(3):
        # SIR_host[i] = 1.0/(1.0 + torch.exp(-self.slope[i]*(SIR_host[i]-self.threshold[i])))
        # SIR_host[i] = 1.0/(1.0 + torch.exp(-self.slope[i]*(UIV_host[i]-self.threshold[i])))
        # SIR_host = self.W(SIR_host)
        return self.W(UIV_host)  # self.W(SIR_host)

    def get_params(self):
        params = dict()
        params['weight'] = self.W.weight.detach().cpu().numpy()
        if hasattr(self.W, 'bias'):
            params['bias'] = self.W.bias.detach().cpu().numpy()
        if hasattr(self, 'slope'):
            for i in range(3):
                params['threshold_'+str(i)] = {'slope': self.slope[i].detach().cpu().numpy(),
                                               'threshold': self.threshold[i].detach().cpu().numpy()}
        return params


class nUIV_NODE(nn.Module):
    def __init__(self, num_hosts: int, **kwargs):
        super().__init__()
        self.num_hosts = torch.tensor(num_hosts)
        self.nUIV_x0 = nn.Parameter(torch.rand(3*self.num_hosts))  # initialize a random initial state
        self.nUIV_dynamics = nUIV_rhs(self.num_hosts)  # initialize a random nUIV
        self.nUIV_to_SIR = soft_threshold()
        self.method = kwargs.pop('method', 'rk4')
        self.step_size = kwargs.pop('step_size', None)

    def simulate(self, times):
        solution = odeint(self.nUIV_dynamics, self.nUIV_dynamics.parametrization(self.nUIV_x0),
                          times, method=self.method, options=dict(step_size=self.step_size)).to(times.device)
        SIR = torch.zeros(3, len(times), device=times.device)
        for t in range(len(times)):
            for i in range(self.num_hosts):
                SIR[:, t] += self.nUIV_to_SIR(solution[t, 3*i:3*i+3])
        return SIR/self.num_hosts

    def get_params(self):
        params = dict()
        with torch.no_grad():
            self.nUIV_dynamics.normalize_ts()
            params['beta'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.betas).detach().cpu().numpy()
            params['delta'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.deltas).detach().cpu().numpy()
            params['p'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.ps).detach().cpu().numpy()
            params['c'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.cs).detach().cpu().numpy()
            params['t'] = self.nUIV_dynamics.parametrization(self.nUIV_dynamics.ts).detach().cpu().numpy()
            params['x0'] = self.nUIV_dynamics.parametrization(self.nUIV_x0).detach().cpu().numpy()
            params['nUIV_to_SIR'] = self.nUIV_to_SIR.get_params()
        return params
