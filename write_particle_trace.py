import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter

fileprefix_snapshot = 'snapdir_%03d/snapshot_%03d'

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot min,max> <ID>')
    return 1

  ID = int(argv[2])

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]

  with open('trace_%d.txt'%(ID),'wt') as f:
    f.write('# snapshot, a, x, y, z\n')
    for ss in range(ssmin,ssmax+1):
      pos, header = read_particles_filter(fileprefix_snapshot%(ss,ss),ID_list=[ID],opts={'pos':True})
      f.write('%d %.6e %.6e %.6e %.6e\n'%(ss, header['Time'], pos[0,0], pos[0,1], pos[0,2]))

  print(pos.tolist())

  return 0

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

