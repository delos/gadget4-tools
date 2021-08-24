import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot> <ID>')
    return 1

  ID = int(argv[2])

  pos, _ = read_particles_filter(argv[1],ID_list=[ID],opts={'pos':True})

  print(pos.tolist())

  return 0

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

