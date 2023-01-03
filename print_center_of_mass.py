import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot> [types] [periodic=1]')
    return 1

  try:
    types = [int(x) for x in argv[2].split(',')]
    if np.any(np.array(types)<0):
      types = None
    #print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None
  
  if len(argv) > 3 and argv[3][0].lower() not in ['f','n','0']:
    periodic = True
  else:
    periodic = False
    #print('nonperiodic')

  pos, vel, mass, header = read_particles_filter(argv[1],type_list=types,opts={'mass':True,'pos':True,'vel':True},verbose=False)
  if periodic:
    box = header['BoxSize']
    shift = pos[0:1] - 0.5*box
    cm = (np.average((pos-shift)%box,weights=mass,axis=0)+shift)%box
  else:
	  cm = np.average(pos,weights=mass,axis=0)
  cmv = np.average(vel,weights=mass,axis=0)

  print('%g %g %g %g %g %g %g'%(header['Time'],cm[0],cm[1],cm[2],cmv[0],cmv[1],cmv[2],))

  return 0

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

