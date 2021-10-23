# Contains items I type to learn how to create my own GAN

import os
import scipy.io as io
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
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
voxels = io.loadmat('./3DShapeNets/volumetric_data/airplane/30/train/3e73b236f62d05337678474be485ca_12.mat')['instance']
voxels = np.pad(voxels, (1,1), 'constant', constant_values = (0,0))
voxels = nd.zoom(voxels, 2, mode='constant', order=0)
print(f'Shape of Voxels: {np.shape(voxels)}')

fig = plt.figure()
ax = fig.add_subplot(projection='3d', aspect='auto')
ax.voxels(voxels, edgecolor="red")
plt.show()