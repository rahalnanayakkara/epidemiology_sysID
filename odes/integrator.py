import numpy as np
from scipy.integrate import solve_ivp


class integrator:
    '''
    Helper class for numerical integration of ODE objects
    is handed an ODE object that it iterates on

    args:
        - ODE : object with a right-hand-side method that takes state as argument
        - x0 : initial state of system
        - dt : time-step to take between "step" calls
        - integration_dt : timesteps for numerical integration
    '''
    def __init__(self, ODE, x0: np.ndarray, dt: float, **kwargs):
        self.ODE = ODE
        self.x = x0
        self.dt = dt
        self.t = 0.0
        self.x0 = x0
        self.method = kwargs.pop('method', None)

    def step(self):

        def f(t, x):
            return self.ODE.RHS(self.x)
        if self.method is None:
            sol = solve_ivp(f, (self.t, self.t + self.dt), self.x)
        else:
            sol = solve_ivp(f, (self.t, self.t + self.dt), self.x, method=self.method)
        self.t += self.dt  # update time
        self.x = sol.y[:, -1]  # update state

        return self.x

    def reset(self, x0=None, t0=None):
        if x0 is not None:
            self.x = x0
        else:
            print('Resetting to stored initial state.')
            self.x = self.x0
        if t0 is not None:
            self.t = t0
        else:
            print('Resetting to t=0.0.')
            self.t = 0.0
        return self.x
