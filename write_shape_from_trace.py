import numpy as np
from numba import njit
from snapshot_functions import list_snapshots, read_particles_filter
from scipy.linalg import eigh

@njit
def inertia_tensor(pos,mass,axes):
  I = np.zeros((3,3))
  M = 0.
  reduced_axes = axes/axes[-1]
  for p in range(pos.shape[0]):
    r2 = np.sum((pos[p]/reduced_axes)**2)
    if r2 > axes[-1]**2:
      continue
    for j in range(3):
      for k in range(3):
        I[j,k] += mass[p] * pos[p,j]*pos[p,k] / r2
    M += mass[p]
  I /= M
  return I, M

def shape(pos,mass,r):
  axes = np.ones(3)*r

  while True:
    I, M = inertia_tensor(pos,mass,axes)
    newaxes, eigvec = eigh(I) # ascending order
    newaxes = np.sqrt(newaxes/newaxes[-1])*r
    if np.sqrt(np.sum((newaxes-axes)**2)) < 1e-7:
      axes = newaxes
      break
    axes = newaxes
    rot = eigvec.T
    #print(rot,newaxes/r)
    pos = (rot@(pos.T)).T

  return axes, M

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot min,max> <trace-file> <r> [types=-1] [out-prefix]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]

  _data = np.loadtxt(argv[2])
  _ss = _data[:,0]
  _x = _data[:,2]
  _y = _data[:,3]
  _z = _data[:,4]
  print('using trace file')
  
  r = float(argv[3])
  print('spherical radius %g'%r)

  try:
    types = [int(x) for x in argv[4].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  
  try: outbase = argv[5]
  except: outbase = 'shape'

  names, headers = list_snapshots()
  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])

    if snapshot_number < ssmin or snapshot_number > ssmax:
      continue

    outname = outbase + filename[-4:] + '.txt'

    if snapshot_number < _ss[0]:
      center = [_x[0],_y[0],_z[0]]
    elif snapshot_number > _ss[-1]:
      center = [_x[-1],_y[-1],_z[-1]]
    else:
      center = [
        _x[_ss==snapshot_number][0],
        _y[_ss==snapshot_number][0],
        _z[_ss==snapshot_number][0],
        ]

    pos, mass, header = read_particles_filter(filename,center=center,radius=r,type_list=types,opts={'mass':True,'pos':True})

    axes, M = shape(pos,mass,r)
    c, b, a = axes

    with open(outname,'at') as f:
      #f.write('# a b c\n')
      f.write('%.6e %.6e %.6e  %.6e\n'%(a, b, c, M))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
