import numpy as np
import matplotlib.pyplot as plt
from snapshot_functions import list_snapshots, read_particles_filter

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <types> [output]')
    return 1

  types = [int(x) for x in argv[1].split(',')]
  if np.any(np.array(types)<0):
    types = None
  print('particle types ' + ', '.join([str(x) for x in types]))

  try: outname = argv[2]
  except: outname = 'trace_type%s.txt'%(argv[1])

  names, headers = list_snapshots()
  l_ss, l_a, l_X, l_V = [], [], [], []

  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])

    pos, vel, mass, header = read_particles_filter(filename,type_list=types,opts={'mass':True,'pos':True,'vel':True})
    box = header['BoxSize']
    Xa = np.average(pos,weights=mass,axis=0)
    Xb = (np.average((pos+box*1./3)%box,weights=mass,axis=0)-box*1./3)%box
    Xc = (np.average((pos+box*2./3)%box,weights=mass,axis=0)-box*2./3)%box
    X = np.median([Xa,Xb,Xc],axis=0)
    V = np.average(vel,weights=mass,axis=0)

    l_ss += [snapshot_number]
    l_a += [header['Time']]
    l_X += [X]
    l_V += [V]
  
  with open(outname,'wt') as f:
    f.write('# snapshot, a, x, y, z, vx, vy, vz\n')
    for i in range(len(l_ss)):
      f.write('%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n'%(
        l_ss[i], l_a[i], l_X[i][0], l_X[i][1], l_X[i][2], l_V[i][0], l_V[i][1], l_V[i][2], ))
  
if __name__ == '__main__':
  from sys import argv
  run(argv)                         
