import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
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
        #self.betas = nn.Parameter(1.35*10**(-7)*torch.ones((self.N,)))
        #self.deltas = nn.Parameter(0.61*torch.ones((self.N,)))
        #self.cs = nn.Parameter(2.4*torch.ones((self.N,)))
        #self.ps = nn.Parameter(0.2*torch.ones((self.N,)))
        self.betas = nn.Parameter(torch.rand((self.N,)))
        self.deltas = nn.Parameter(torch.rand((self.N,)))
        self.cs = nn.Parameter(torch.rand((self.N,)))
        self.ps = nn.Parameter(torch.rand((self.N,)))
        self.ts = nn.Parameter(torch.eye(self.N) + torch.rand((self.N, self.N))/2.0)
        self.parametrization = squared_parametrization()
        #self.parametrization = abs_parametrization()
        P.register_parametrization(self, 'betas', self.parametrization)
        P.register_parametrization(self, 'deltas', self.parametrization)
        P.register_parametrization(self, 'cs', self.parametrization)
        P.register_parametrization(self, 'ps', self.parametrization)
        P.register_parametrization(self, 'ts', self.parametrization)

    def forward(self, t, state):
        rhs = torch.zeros_like(state)  # (U, I, V) RHS's for each node
        normalization = self.compute_normalization()
        rhs[::3] = -self.betas*state[::3]*state[2::3]
        rhs[1::3] = self.betas*state[::3]*state[2::3] - self.deltas*state[1::3]
        rhs[2::3] = self.ps*state[1::3] - self.cs*state[2::3] \
            - (1.0 + normalization)*torch.diag(self.ts)*state[2::3]
        rhs[2::3] += torch.matmul(state[2::3].T, torch.matmul(torch.diag(normalization), self.ts))
        return rhs

    def compute_normalization(self):
        normalization = torch.zeros_like(self.betas)
        normalization = torch.min(torch.diag(self.ts) / (torch.sum(self.ts, axis=1)
                                  - torch.diag(self.ts)), torch.tensor(1.0))
        return normalization

    def normalize_ts(self):
        normalization = self.compute_normalization()
        for n in range(self.N):
            self.ts[n, :n] *= normalization[n]
            self.ts[n, n+1:] *= normalization[n]


class soft_threshold(nn.Module):
    '''
    function for mapping from a single host's UIV state to their SIR state.
    Currently defined as a linear map plus a soft-thresholding operator.
    '''
    def __init__(self, bias=False, nonlinearity=False):
        super().__init__()
        self.W = nn.Linear(3, 3, bias=bias)  # linear map
        if nonlinearity:
            #self.slope = nn.ParameterList([torch.tensor(10.0)+torch.rand(1) for i in range(3)])
            #self.threshold = nn.ParameterList([torch.rand(1) for i in range(3)])
            self.slope = nn.Parameter(torch.tensor(10.0)+torch.rand(1))
            self.threshold = nn.Parameter(torch.rand(1))

    #def forward(self, UIV_host):
        # SIR_host = UIV_host  # self.W(UIV_host)
        #SIR_host = torch.zeros(3, device=UIV_host.device)
        # for i in range(3):
        # SIR_host[i] = 1.0/(1.0 + torch.exp(-self.slope[i]*(SIR_host[i]-self.threshold[i])))
        # SIR_host[i] = 1.0/(1.0 + torch.exp(-self.slope[i]*(UIV_host[i]-self.threshold[i])))
        # SIR_host = self.W(SIR_host)
        #return self.W(UIV_host)  # self.W(SIR_host)
    
    def forward(self,UIV_host):
        SIR_host = UIV_host
        SIR_host[0] = 0
        SIR_host[2] = 0
        I_host = 1.0/(1.0 + torch.exp(-self.slope*(UIV_host[2]-self.threshold)))
        SIR_host[1] = I_host
        return SIR_host
    '''
    def get_params(self):
        params = dict()
        params['weight'] = self.W.weight.detach().cpu().numpy()
        if self.W.bias is not None:
            params['bias'] = self.W.bias.detach().cpu().numpy()
        if hasattr(self, 'slope'):
            for i in range(3):
                params['threshold_'+str(i)] = {'slope': self.slope[i].detach().cpu().numpy(),
                                               'threshold': self.threshold[i].detach().cpu().numpy()}
        return params
    '''
    def get_params(self):
        params = dict()
        params['slope'] = self.slope.detach().cpu().numpy()
        params['threshold'] = self.threshold.detach().cpu().numpy()
        return params


class nUIV_NODE(nn.Module):
    def __init__(self, num_hosts: int, **kwargs):
        super().__init__()
        self.num_hosts = torch.tensor(num_hosts)
        self.nUIV_x0 = nn.Parameter(torch.rand(3*self.num_hosts))  # initialize a random initial state
        #tensor = torch.zeros(3 * self.num_hosts)
        #exp_values = torch.normal(mean=0.0, std=0.05, size=(self.num_hosts,))
        #tensor[::3] = 10**(9 + exp_values)
        #tensor[::3] = 10**9*torch.ones((self.num_hosts,))
        #tensor[2::3] = 10*torch.ones((self.num_hosts,))
        #self.nUIV_x0 = nn.Parameter(tensor)
        self.nUIV_dynamics = nUIV_rhs(self.num_hosts)  # initialize a random nUIV
        self.nUIV_to_SIR = soft_threshold(nonlinearity=1)

        self.parametrization = squared_parametrization()
        #self.parametrization = abs_parametrization()
        P.register_parametrization(self, 'nUIV_x0', self.parametrization)

        self.method = kwargs.pop('method', 'rk4')
        self.step_size = kwargs.pop('step_size', None)

    def _create_custom_tensor(self):
        tensor = torch.zeros(3 * self.num_hosts)
        exp_values = torch.normal(mean=0.0, std=0.5, size=(self.num_hosts,))
        tensor[::3] = 10 ** (9 + exp_values)
        tensor[2::3] = torch.normal(mean=10.0, std=0.5, size=(self.num_hosts,))
        return tensor

    def simulate(self, times):
        solution = odeint(self.nUIV_dynamics, self.nUIV_x0,
                          times, method=self.method, options=dict(step_size=self.step_size)).to(times.device)
        #SIR = torch.zeros(3, len(times), device=times.device)
        if torch.isnan(solution).any():
            print("Cannot solve the ODEs!")
            print(solution)
        UIV_initial = torch.reshape(solution, (len(times), self.num_hosts, 3))
        SIR_initial = self.nUIV_to_SIR(UIV_initial)
        SIR = torch.sum(SIR_initial, axis=1).T
        #SIR_normal = SIR_initial/SIR_initial.sum(dim=2, keepdim=True)
        #SIR = torch.sum(SIR_normal, axis=1).T
        return SIR/self.num_hosts
        #return I_sum/self.num_hosts

    def get_params(self):
        params = dict()
        with torch.no_grad():
            self.nUIV_dynamics.normalize_ts()
            params['beta'] = self.nUIV_dynamics.betas.detach().cpu().numpy()
            params['delta'] = self.nUIV_dynamics.deltas.detach().cpu().numpy()
            params['p'] = self.nUIV_dynamics.ps.detach().cpu().numpy()
            params['c'] = self.nUIV_dynamics.cs.detach().cpu().numpy()
            params['t'] = self.nUIV_dynamics.ts.detach().cpu().numpy()
            params['x0'] = self.nUIV_x0.detach().cpu().numpy()
            params['nUIV_to_SIR'] = self.nUIV_to_SIR.get_params()
        return params
