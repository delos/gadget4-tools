import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter, read_subhalos

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <group-file> [output-file]')
    return 1
  
  try: outbase = argv[2]
  except: outbase = 'subhalos.txt'

  # read halos
  hpos, hmas, header = read_subhalos(argv[1],opts={'pos':True,'mass':True,})
  NH = hmas.size

  print('%d subhalos'%NH)

  np.savetxt('subhalos.txt',np.concatenate((hmas[:,None],hpos),axis=1),header='%.12e'%header['Time'])

if __name__ == '__main__':
  from sys import argv
  run(argv)
