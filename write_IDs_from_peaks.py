import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter
from scipy.spatial import cKDTree

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <peaks-file,grid-size> <preIC-file> [num particles=7] [types] [dist/grid] [out-file]')
    return 1

  peakfile, GridSize = argv[1].split(',')
  GridSize = int(GridSize)

  snapshot = argv[2]

  outfile = 'peak_particles.txt'
  if len(argv) > 6:
    outfile = argv[6]
  print('will write to %s'%outfile)

  try: npart = int(argv[3])
  except: npart = 7

  try:
    types = [int(x) for x in argv[4].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  dlim = np.inf
  if len(argv) > 5:
    dlim = float(argv[5])
    print('distance limit = %f cells'%dlim)

  pos, ID, header = read_particles_filter(snapshot,type_list=types,opts={'pos':True,'vel':False,'ID':True,'mass':False})

  num, i, j, k = np.loadtxt(peakfile,usecols=(0,1,2,3)).astype(int).T
  peakpos = (np.array([i,j,k]).T + 0.5) * header['BoxSize'] / GridSize

  print('finished reading')

  tree = cKDTree(pos)

  print('created tree')

  dist,index = tree.query(peakpos,npart,distance_upper_bound=dlim*header['BoxSize']/GridSize) # peakpos shape (Npeak, 3) -> index shape (Npeak,npart)

  print('writing')

  with open(outfile,'wt') as f:
    for p,n in enumerate(num):
      if np.any(np.isinf(dist[p])):
        continue
      f.write('%d '%n)
      for idx in index[p]:
        f.write(' %d'%ID[idx])
      f.write('\n')

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

