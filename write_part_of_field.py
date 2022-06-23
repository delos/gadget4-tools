import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = cm.viridis

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <x,y,z,r>')
    return 1
  
  filename = argv[1]
  
  x,y,z,r = [int(val) for val in argv[2].split(',')]

  xlim = np.array([x-r,x+r])
  ylim = np.array([y-r,y+r])
  zlim = np.array([z-r,z+r])

  dsize = os.stat(filename).st_size//4
  GridSize = int(dsize**(1./3)+.5)

  with open(filename,'rb') as f:
    f.seek(GridSize**2*xlim[0]*4)
    delta = np.fromfile(f,count=GridSize**2*(xlim[1]-xlim[0]+1),dtype=np.float32)
  delta.shape = (xlim[1]-xlim[0]+1,GridSize,GridSize)

  delta = delta[:,ylim[0]:ylim[1]+1,zlim[0]:zlim[1]+1]
  
  delta.tofile(filename + '_%d_%d_%d_%d'%(x,y,z,r))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
