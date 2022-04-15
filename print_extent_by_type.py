import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter
from pathlib import Path
import h5py


def run(argv):
  
  if len(argv) < 2:
    print('python script.py <snapshot> [types=-1]')
    return 1

  filename = argv[1]
  
  try:
    types = [int(x) for x in argv[2].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except Exception as e:
    print('all types (%s)'%str(e))
    types = None

  # read
  pos, header = read_particles_filter(filename,type_list=types,opts={'pos':True})
  print('%d particles'%(pos.shape[0]))

  xmin = np.min(pos,axis=0)/header['BoxSize']
  xmax = np.max(pos,axis=0)/header['BoxSize']
  print('minimum: (' + ','.join(['%.3f'%x for x in xmin]) + ') box')
  print('maximum: (' + ','.join(['%.3f'%x for x in xmax]) + ') box')
  print(' extent: (' + ','.join(['%.3f'%x for x in xmax-xmin]) + ') box')

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
