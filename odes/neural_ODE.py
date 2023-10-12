import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint


class squared_parametrization(nn.Module):
    def forward(self, W):
        return torch.square(W)

    def right_inverse(self, W):
        return torch.sqrt(W)


class abs_parametrization(nn.Module):
    def forward(self, W):
        return torch.abs(W)

    def right_inverse(self, W):
        return W


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
        self.betas = nn.Parameter(torch.rand((self.N,)))
        self.deltas = nn.Parameter(torch.rand((self.N,)))
        self.cs = nn.Parameter(torch.rand((self.N,)))
        self.ps = nn.Parameter(torch.rand((self.N,)))
        self.ts = nn.Parameter((torch.eye(self.N) + torch.rand((self.N, self.N)))/2.0)
        self.parametrization = squared_parametrization()
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
        rhs[2::3] += torch.matmul(state[2::3], torch.matmul(torch.diag(normalization), self.ts))
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
        if nonlinearity:
            self.slope = nn.Parameter(torch.tensor(1.0))  # *torch.ones(3)
            self.threshold = nn.Parameter(torch.rand(3)*200)  # nn.Parameter(torch.rand(3))
            self.act = nn.Sigmoid()
            self.W = nn.Linear(3, 3, bias=bias)
            self.forward = self.forward_nonlinear
        else:
            self.W = nn.Linear(3, 3, bias=bias)  # linear map
            self.forward = self.forward_linear

    def forward_linear(self, UIV_host):
        return torch.nn.functional.normalize(self.W(UIV_host), p=1, dim=-1)

    def forward_nonlinear(self, UIV_host):
        return torch.nn.functional.normalize(torch.sigmoid(self.slope*(UIV_host-self.threshold)), p=1, dim=-1)
        # return self.act(UIV_host-self.threshold)  # 1.0/(1.0 + torch.exp(-self.slope*(UIV_host - self.threshold)))

    def get_params(self):
        params = dict()
        if hasattr(self, 'W'):
            params['weight'] = self.W.weight.detach().cpu().numpy()
            if self.W.bias is not None:
                params['bias'] = self.W.bias.detach().cpu().numpy()
        if hasattr(self, 'slope'):
            params['slope'] = self.slope.detach().cpu().numpy()
        if hasattr(self, 'threshold'):
            params['threshold'] = self.threshold.detach().cpu().numpy()
            # for i in range(3):
            #     params['threshold_'+str(i)] = {'slope': self.slope[i].detach().cpu().numpy(),
            #                                    'threshold': self.threshold[i].detach().cpu().numpy()}
        return params


class nUIV_NODE(nn.Module):
    def __init__(self, num_hosts: int, **kwargs):
        super().__init__()
        self.num_hosts = torch.tensor(num_hosts)
        self.nUIV_x0 = nn.Parameter(torch.rand(3*self.num_hosts))  # initialize a random initial state
        self.nUIV_dynamics = nUIV_rhs(self.num_hosts)  # initialize a random nUIV
        self.nUIV_to_SIR = soft_threshold(nonlinearity=kwargs.pop('nonlinearity', False))

        self.parametrization = squared_parametrization()
        P.register_parametrization(self, 'nUIV_x0', self.parametrization)

        self.method = kwargs.pop('method', 'rk4')
        self.step_size = kwargs.pop('step_size', None)

    def simulate(self, times):
        solution = odeint(self.nUIV_dynamics, self.nUIV_x0,
                          times, method=self.method, options=dict(step_size=self.step_size)).to(times.device)  # shape T x 3*num_hosts
        # SIR = torch.zeros(3, len(times), device=times.device)
        SIR = torch.mean(self.nUIV_to_SIR(torch.reshape(solution, (len(times), self.num_hosts, 3))), axis=1).transpose(0, 1)  # result should be 3 x T
        return SIR

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
