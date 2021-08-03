import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = cm.viridis

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <filename> [projection axis=2]')
    return 1
  
  filename = argv[1]
  
  if len(argv) > 2: axis = int(argv[2])
  else: axis = 2

  dsize = os.stat(filename).st_size//4
  GridSize = int(dsize**(1./3)+.5)
  with open(filename,'rb') as f:
    delta = np.fromfile(f,count=-1,dtype=np.float32)
  delta.shape = (GridSize,GridSize,GridSize)

  delta = np.mean(delta,axis=axis) # project

  outname = filename + '.%d.2d'%axis
  delta.tofile(outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
