import numpy as np
import matplotlib.pyplot as plt
from snapshot_functions import gadget_to_particles_DMO, cic_bin, power_spectrum

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <grid size>')
    return 1
  
  GridSize = int(argv[2])
  
  pos, vel, mass, header = gadget_to_particles_DMO(argv[1])
  
  BoxSize = header['BoxSize']
  
  delta, bins = cic_bin(pos,BoxSize,GridSize,weights=mass,density=True)

  delta /= delta.mean()
  delta -= 1
  
  k,P = power_spectrum(delta,BoxSize)
  
  ax = plt.figure().gca()
  
  ax.loglog(k,k**3/(2*np.pi**2)*P)
  ax.set_xlim(k[0],k[-1])
  ax.set_xlabel(r'$k (h/kpc)$')
  ax.set_ylabel(r'dimensionless $\mathcal{P}(k)$')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
