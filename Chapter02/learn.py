# Contains items I type to learn how to create my own GAN

import os
import scipy.io as io
import numpy as np
from os import system, name  

def clear():
  
    # for windows
    if name == 'nt':
        _ = system('cls')
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

clear()
print(f'Current Working Directory: {os.getcwd()}')
voxels = io.loadmat('./3DShapeNets/volumetric_data/door/30/train/door_000001796_12.mat')['instance']
print(f'Shape of Voxels: {np.shape(voxels)}')
pass