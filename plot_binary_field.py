import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = cm.viridis

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <filename> [projection axis=2] [log=1]')
    return 1
  
  filename = argv[1]
  
  if len(argv) > 2: axis = int(argv[2])
  else: axis = 2
  if len(argv) > 3: log = int(argv[3])
  else: log = 1

  dsize = os.stat(filename).st_size//4
  GridSize = int(dsize**(1./3)+.5)
  with open(filename,'rb') as f:
    delta = np.fromfile(f,count=-1,dtype=np.float32)
  delta.shape = (GridSize,GridSize,GridSize)
  bins = np.arange(GridSize+1)

  print('max = %f'%delta.max())
  print('min = %f'%delta.min())
  print('sigma = %f'%(np.mean(delta**2)**0.5))

  delta = np.mean(delta,axis=axis) # project

  if log:
    np.log10(1+delta,out=delta)
  
  fig, ax = plt.subplots(figsize=(12., 10))
  im = ax.imshow(delta.T,origin='lower')
  cbar = fig.colorbar(im, ax=ax)
  ax.patch.set_facecolor('k')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
