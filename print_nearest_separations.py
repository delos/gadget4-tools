import numpy as np
from scipy.spatial import cKDTree as KDTree
from numba import njit
from snapshot_functions import read_particles_filter

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <snapshot>')
    return 1

  pos, header = read_particles_filter(argv[1],opts={'pos':True}, verbose = False)

  tree = KDTree(pos)
  minsep = tree.query(pos,2,)
  print('%g  %g'%(header['Time'],minsep[0][:,1].min()))

  return 0

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

