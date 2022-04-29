import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = cm.viridis

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <filename> [projection axis=2] [project square=0]')
    return 1
  
  filename = argv[1]
  
  if len(argv) > 2: axis = int(argv[2])
  else: axis = 2

  if len(argv) > 3: sq = int(argv[3])
  else: sq = 0

  dsize = os.stat(filename).st_size//4
  GridSize = int(dsize**(1./3)+.5)
  with open(filename,'rb') as f:
    delta = np.fromfile(f,count=-1,dtype=np.float32)
  delta.shape = (GridSize,GridSize,GridSize)

  if sq:
    delta = np.sqrt(np.mean(delta**2,axis=axis))
  else:
    delta = np.mean(delta,axis=axis) # project

  if axis == 1:
    delta = delta.T # maintain parity

  outname = filename + '.%d.2d'%axis
  delta.tofile(outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
