import numpy as np
import networkx as nx


class SIR:
    '''
    SIR model object.
    Holds an ODE defining bulk population statistics, i.e.
    S - fraction of population that is susceptible
    I - fraction of population that is infect
    R - fraction of population that is recovering

    args:
        - beta : infection rate between infected and susceptible population
        - gamma : recovery rate from infected to recovered populations
        - N : number of people

    population statistics obey the dynamics:

    dS/dt = -beta * I * S / N
    dI/dt = beta * I * S / N - gamma * I
    dR/dt = gamma * I
    '''
    def __init__(self, N: int, beta: float, gamma: float):
        self.N = float(N)
        self.beta = beta
        self.gamma = gamma

    def RHS(self, state):
        rhs = np.empty(3,)
        S, I, _ = state[0], state[1], state[2]  # renaming for clarity

        # compute SIR model ODE RHS
        rhs[0] = -self.beta*S*I/self.N
        rhs[1] = self.beta*S*I/self.N - self.gamma*I
        rhs[2] = self.gamma*I
        return rhs


class nUIV:
    '''
    networked UIV model object
    holds an ODE defining a network of in-host models.
    args:
        - G : networkx graph defining the relationship betwen the hosts.
              G must have edges with attribute "transmit"
              G must have nodes with attributes:
                    - "state" (U, I, V)
                    - "beta" - infection rate of healthy cells
                    - "delta" - death rate of infected cells
                    - "p" - virus replication rate
                    - "c" - clearance rate of virus cells
    '''
    def __init__(self, G: nx.Graph):
        self.G = G
        self.N = self.G.number_of_nodes()
        self.statekey = 'state'
        self.betakey = 'beta'
        self.deltakey = 'delta'
        self.pkey = 'p'
        self.ckey = 'c'
        self.tkey = 'transmit'

    def RHS(self, state=None):
        rhs = np.empty(3 * self.N)  # (U, I, V) RHS's for each node
        # for each host
        if state is not None:
            self.set_graph_state(state)
        for (i, n) in enumerate(self.G):
            U, I, V = self.G.nodes[n][self.statekey]  # re-naming host state for convenience
            beta = self.G.nodes[n][self.betakey]
            delta = self.G.nodes[n][self.deltakey]
            p = self.G.nodes[n][self.pkey]
            c = self.G.nodes[n][self.ckey]

            rhs[3*i] = - beta * U * V  # U dynamics
            rhs[3*i + 1] = beta * U * V - delta * I  # I dynamics
            rhs[3*i + 2] = p*I - c*V  # V dynamics

            # add transmission between neighbors (including self)
            for nbr in self.G.predecessors(n):
                rhs[3*i + 2] += self.G[nbr][n][self.tkey]*self.G.nodes[nbr][self.statekey][2]

        return rhs

    def set_graph_state(self, state):
        '''
        Applies the given state the networkX graph instance nodes.
        args:
            - state (np.ndarray) : shape (3*N) array of UIV states for each node stacked
        returns:
        '''
        for (i, n) in enumerate(self.G):
            self.G.nodes[n][self.statekey] = np.array([state[3*i], state[3*i + 1], state[3*i + 2]])

    def get_graph_state(self):
        '''
        pops the current state from the graph into a numpy array
        '''
        state_numpy = np.zeros(self.N*3)
        for i, (node, node_data) in enumerate(self.G.nodes.items()):
            state_numpy[3*i:3*i+3] = node_data[self.statekey]
        return state_numpy
