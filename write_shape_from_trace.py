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
    if M == 0. or not np.isfinite(M) or not np.all(np.isfinite(I)):
      return np.zeros(3), 0.
    newaxes, eigvec = eigh(I) # ascending order
    if np.any(newaxes<0.):
      return np.zeros(3), 0.
    newaxes = np.sqrt(newaxes/newaxes[-1])*r
    diff = np.sqrt(np.sum((newaxes-axes)**2))/r
    print(axes/r, diff)
    if diff < 1e-3:
      axes = newaxes
      break
    axes = newaxes
    rot = eigvec.T
    #print(rot,axes/r)
    pos = (rot@(pos.T)).T

  return axes, M

@njit
def cull(pos,mass,r):
  q = 0
  for p in range(pos.shape[0]):
    if np.sum(pos[p]**2) <= r**2:
      pos[q] = pos[p]
      mass[q] = mass[p]
      q += 1
  return q

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot min,max> <trace-file> <rmin,rmax,nr> [types=-1] [out-prefix]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]

  _data = np.loadtxt(argv[2])
  _ss = _data[:,0]
  _x = _data[:,2]
  _y = _data[:,3]
  _z = _data[:,4]
  print('using trace file')
  
  rb = argv[3].split(',')
  rmin = float(rb[0])
  rmax = float(rb[1])
  nr = int(rb[2])
  print('%d radii from %g to %g'%(nr,rmin,rmax))

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

    pos, mass, header = read_particles_filter(filename,center=center,radius=rmax,type_list=types,opts={'mass':True,'pos':True})
    size = mass.size
    with open(outname,'wt') as f:
      f.write('# a b c  M\n')
      for j,r in enumerate(np.geomspace(rmin,rmax,nr)[::-1]):
        size = cull(pos[:size],mass[:size],r)
        print('r=%g, N=%d'%(r,size))
        axes, M = shape(pos[:size],mass[:size],r)
        c, b, a = axes
        string = '%.4e %.4e %.4e  %.6e'%(a, b, c, M)
        f.write(string + '\n')
        print(string)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
