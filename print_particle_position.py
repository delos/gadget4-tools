import numpy as np
from numba import njit
from snapshot_functions import gadget_to_particles

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot> <ID>')
    return 1

  ID1 = int(argv[2])

  pos, ID, _ = gadget_to_particles(argv[1],opts={'pos':True,'vel':False,'ID':True,'mass':False})

  idx = np.where(ID == ID1)[0]

  if len(idx) > 0:
    idx = idx[0]
  else:
    print('not found')
    return 1

  print(list(pos[:,idx]))

  return 0

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

