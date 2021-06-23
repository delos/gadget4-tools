import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IC_functions import gadget_to_particles_DMO, cic_bin

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <grid size> [projection axis=2] [log=1]')
    return 1
  
  GridSize = int(argv[2])
  
  if len(argv) > 3: axis = int(argv[3])
  else: axis = 2
  if len(argv) > 4: log = int(argv[4])
  else: log = 1
  
  pos, vel, mass, header = gadget_to_particles_DMO(argv[1])
  
  BoxSize = header['BoxSize']
  
  delta, bins = cic_bin(pos,BoxSize,GridSize,weights=mass,density=True)
  
  if log:
    delta = np.log10(1+delta)
  
  fig, ax = plt.subplots(figsize=(12.9, 10))
  cmap = cm.viridis
  pcm = ax.pcolormesh(bins,bins,delta,cmap=cmap)
  cbar = fig.colorbar(pcm, ax=ax)
  ax.set_aspect('equal', 'datalim')
  ax.set_xlim(0,ncell)
  ax.set_ylim(0,ncell)
  ax.patch.set_facecolor('k')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
