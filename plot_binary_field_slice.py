import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = cm.viridis

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <axis,min,max> [log=1]')
    return 1
  
  filename = argv[1]
  
  axis = argv[2].split(',')
  amin = int(axis[1])
  amax = int(axis[2])
  axis = int(axis[0])

  if len(argv) > 3: log = int(argv[3])
  else: log = 1

  dsize = os.stat(filename).st_size//4
  GridSize = int(dsize**(1./3)+.5)
  with open(filename,'rb') as f:
    delta = np.fromfile(f,count=-1,dtype=np.float32)
  delta.shape = (GridSize,GridSize,GridSize)
  bins = np.arange(GridSize+1)

  if axis == 0:
    delta = np.mean(delta[amin:amax+1],axis=axis)
  if axis == 1:
    delta = np.mean(delta[:,amin:amax+1],axis=axis)
  if axis == 2:
    delta = np.mean(delta[:,:,amin:amax+1],axis=axis)

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
