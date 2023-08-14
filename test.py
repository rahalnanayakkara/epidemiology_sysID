import os
import pickle
import numpy as np

path = './tmp/successful_runs/run03/'
filename = os.path.join(path, 'params.p')
with open(filename, 'rb') as f:
    sim_params = pickle.load(f)

nUIV_params = sim_params['nUIV']
SIR_params = sim_params['SIR']

np.set_printoptions(threshold=np.inf)
filename = os.path.join(path, 'params.txt')
with open(filename, 'w') as f:
    f.write('SIMULATION PARAMATERS:\n')
    for key, value in sim_params.items():
        f.write(f'{key} : {value}\n')
    f.write('SIR MODEL PARAMETERS\n')
    for key, value in SIR_params.items():
        f.write(f'{key} : {value}\n')
    f.write('NUIV MODEL PARAMETERS\n')
    for key, value in nUIV_params.items():
        f.write(f'{key} : {value}\n')
