import numpy as np
import matplotlib.pyplot as plt
from snapshot_functions import fof_to_halos

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <fof-file>')
    return 1
  
  # read halos
  _, _, hmass, header = fof_to_halos(argv[1])
  #BoxSize = header['BoxSize']

  # sort halos, highest mass first
  sort = np.argsort(hmass)[::-1]
  hmass = hmass[sort]
  
  ax = plt.figure().gca()

  ax.loglog(hmass,np.arange(len(hmass))+1)

  ax.set_xlabel(r'$M$ ($M_\odot/h$)')
  ax.set_ylabel(r'$N(>M)$')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
