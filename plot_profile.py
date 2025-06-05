import numpy as np
import matplotlib.pyplot as plt
from snapshot_functions import gadget_to_particles, density_profile

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot-file> <x,y,z> [scaling index]')
    return 1
  
  hpos = np.array([float(x) for x in argv[2].split(',')])

  # read particles and halos
  pos, _, mass, header = gadget_to_particles(argv[1])
  
  r,rho = density_profile(pos-hpos[None].T,mass)

  try:
    rscale = float(argv[3])
  except:
    rscale = 0.

  ax = plt.figure().gca()
  ax.loglog(r,rho*r**rscale)
  ax.set_xlabel(r'$r$')
  ax.set_ylabel(r'$r^{%g}\rho$'%rscale)
  plt.show()
  ax.figure.savefig('profile.png')

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
