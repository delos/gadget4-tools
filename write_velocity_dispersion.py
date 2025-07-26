import numpy as np
from numba import njit
from snapshot_functions import list_snapshots, read_header, read_particles_filter

def run(argv):
  
  if len(argv) < 1:
    print('python script.py [types=-1]')
    return 1

  try:
    types = [int(x) for x in argv[1].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
    outname = 'velocity_dispersion_' + '_'.join([str(x) for x in types]) + '.txt'
  except:
    types = None
    outname = 'velocity_dispersion.txt'

  names, headers = list_snapshots()

  numbers = []
  times = []
  vdisp = []
  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])
    vel, mass, header = read_particles_filter(filename,type_list=types,opts={'mass':True,'vel':True})
    vdisp += [np.sqrt(np.sum(vel**2 * mass[:,None]) / np.sum(mass*3))]
    numbers += [snapshot_number]
    times += [headers[i]['Time']]

  np.savetxt(outname,np.stack((numbers,times,vdisp)).T,fmt='%.0f %.7e %.7e')

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
