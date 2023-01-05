import numpy as np
import matplotlib.pyplot as plt
from snapshot_functions import read_subhalos

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <fof-file>')
    return 1
  
  # read halos
  mass, somass, header = read_subhalos(argv[1],opts={},
    group_opts={'mass':True,'somass':True},sodef='Mean200')
  #BoxSize = header['BoxSize']
  
  ax = plt.figure().gca()

  ax.loglog(np.sort(mass)[::-1],np.arange(len(mass))+1,label='FOF mass')
  ax.loglog(np.sort(somass)[::-1],np.arange(len(somass))+1,label='SO mass')

  ax.set_xlabel(r'$M$ ($M_\odot/h$)')
  ax.set_ylabel(r'$N(>M)$')

  ax.legend(loc='lower left')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
