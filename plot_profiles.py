import numpy as np
import matplotlib.pyplot as plt
from snapshot_functions import gadget_to_particles, density_profile, fof_to_halos

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot-file> <fof-file> <number of halos>')
    return 1
  
  num_halos = int(argv[3])

  # read particles and halos
  pos, _, mass, header = gadget_to_particles(argv[1])
  BoxSize = header['BoxSize']
  hpos, _, hmass, _ = fof_to_halos(argv[2])

  # sort halos, highest mass first
  sort = np.argsort(hmass)[::-1]
  hpos = hpos[sort]
  hmass = hmass[sort]
  
  ax = plt.figure().gca()

  # cycle over halos
  for i in range(min(num_halos,len(hmass))):
    r,rho = density_profile(pos-hpos[i:i+1].T,mass,BoxSize=BoxSize)
    ax.loglog(r,rho)

  ax.set_xlabel(r'$r$ (kpc/$h$)')
  ax.set_ylabel(r'$\rho$ ($h^2$ $M_\odot$/kpc$^3$)')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
