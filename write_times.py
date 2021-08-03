import numpy as np
from numba import njit
from snapshot_functions import list_snapshots

def run(argv):
  
  if len(argv) < 1:
    print('python script.py')
    return 1

  names, headers = list_snapshots()

  numbers = []
  times = []

  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])

    numbers += [snapshot_number]
    times += [headers[i]['Time']]

  np.savetxt('times.txt',np.stack((numbers,times)).T,fmt='%.0f %.7e')

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
