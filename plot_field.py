import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from snapshot_functions import gadget_to_particles, cic_bin

cmap = cm.viridis

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <grid size> [projection axis=2] [log=1]')
    return 1
  
  GridSize = int(argv[2])
  
  if len(argv) > 3: axis = int(argv[3])
  else: axis = 2
  if len(argv) > 4: log = int(argv[4])
  else: log = 1

  pos, mass, header = gadget_to_particles(argv[1],
    opts={'pos':True,'vel':False,'ID':False,'mass':True})
  
  BoxSize = header['BoxSize']
  
  delta, bins = cic_bin(pos,BoxSize,GridSize,weights=mass,density=True)

  delta /= delta.mean()
  delta -= 1

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
