import numpy as np
from numba import njit
from snapshot_functions import list_snapshots, read_particles_filter
from pathlib import Path
import h5py
from scipy.spatial import cKDTree

# memory usage is 16 bytes per particle + 4 bytes per cell
# more for NN

# use loops for memory-efficiency
@njit
def cic_bin(x,BoxSize,GridSize,weights,density):
  NP = x.shape[0]

  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)

  hist = np.zeros((N,N,N),dtype=np.float32)
  for p in range(NP):
    f = x[p] / dx # (3,)

    i = np.floor(f-0.5).astype(np.int32)
    i1 = i+1

    for d in range(3):
      f[d] -= i[d] + 0.5

    if  i[0] >= 0 and  i[0] < N and  i[1] >= 0 and  i[1] < N and  i[2] >= 0 and  i[2] < N:
      hist[i[0],i[1],i[2]] += (1-f[0])*(1-f[1])*(1-f[2])*weights[p]
    if i1[0] >= 0 and i1[0] < N and  i[1] >= 0 and  i[1] < N and  i[2] >= 0 and  i[2] < N:
      hist[i1[0],i[1],i[2]] += f[0]*(1-f[1])*(1-f[2])*weights[p]
    if  i[0] >= 0 and  i[0] < N and i1[1] >= 0 and i1[1] < N and  i[2] >= 0 and  i[2] < N:
      hist[i[0],i1[1],i[2]] += (1-f[0])*f[1]*(1-f[2])*weights[p]
    if  i[0] >= 0 and  i[0] < N and  i[1] >= 0 and  i[1] < N and i1[2] >= 0 and i1[2] < N:
      hist[i[0],i[1],i1[2]] += (1-f[0])*(1-f[1])*f[2]*weights[p]
    if i1[0] >= 0 and i1[0] < N and i1[1] >= 0 and i1[1] < N and  i[2] >= 0 and  i[2] < N:
      hist[i1[0],i1[1],i[2]] += f[0]*f[1]*(1-f[2])*weights[p]
    if  i[0] >= 0 and  i[0] < N and i1[1] >= 0 and i1[1] < N and i1[2] >= 0 and i1[2] < N:
      hist[i[0],i1[1],i1[2]] += (1-f[0])*f[1]*f[2]*weights[p]
    if i1[0] >= 0 and i1[0] < N and  i[1] >= 0 and  i[1] < N and i1[2] >= 0 and i1[2] < N:
      hist[i1[0],i[1],i1[2]] += f[0]*(1-f[1])*f[2]*weights[p]
    if i1[0] >= 0 and i1[0] < N and i1[1] >= 0 and i1[1] < N and i1[2] >= 0 and i1[2] < N:
      hist[i1[0],i1[1],i1[2]] += f[0]*f[1]*f[2]*weights[p]

  if density:
    hist /= dx**3

  return hist,bins

def nn_density(x,BoxSize,GridSize,weights,nnk,simple=False):
  NP = x.shape[0]

  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)

  hist = np.zeros((N,N,N),dtype=np.float32)

  if NP < nnk:
    return hist,bins

  tree = cKDTree(x / dx - 0.5)
  # particles in (-0.5,N-0.5)
  # put on grid 0,...,N-1

  for i in range(GridSize):
    if i%(GridSize//8) == 0:
      print('%d/%d'%(i,GridSize))
    grid = np.stack([np.full((GridSize,GridSize),i)]+np.meshgrid(*([np.arange(GridSize)]*2),indexing='ij'),axis=-1)
    # grid has shape (GridSize,GridSize,3)
    dist,index = tree.query(grid,nnk)
    # dist, index have shape (GridSize,GridSize,nnk)
    if simple:
      hist[i] = np.sum(weights[index],axis=-1) / (4./3*np.pi*dist.take(-1,axis=-1)**3) # M/V
    else:
      hist[i] = nnk*(nnk+1.)/2. / (4./3*np.pi*np.sum(dist**3/weights[index],axis=-1)) # weighted M/V

  hist /= dx**3
  return hist,bins

def run(argv):
  
  if len(argv) < 5:
    print('python script.py <snapshot min,max> <grid size[, NN k,simple=0]> <x,y,z or trace-file> <r> [types=-1] [rotation-file=0] [ID-file=0] [physical=0] [out-name]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]
  
  method = argv[2].split(',')
  if len(method) == 1:
    GridSize = int(argv[2])
    nnk = None
    nns = False
    print('%d^3 cells; CIC'%(GridSize))
  elif len(method) == 2 or method[2].lower().startswith(('0','f','n','-')):
    GridSize,nnk = [int(x) for x in method[:2]]
    nns = False
    print('%d^3 cells; k=%d nearest neighbors'%(GridSize,nnk))
  else:
    GridSize,nnk = [int(x) for x in method[:2]]
    nns = True
    print('%d^3 cells; k=%d nearest neighbors [simple]'%(GridSize,nnk))

  try:
    _data = np.loadtxt(argv[3])
    _ss = _data[:,0]
    _x = _data[:,2]
    _y = _data[:,3]
    _z = _data[:,4]
    print('using trace file')
  except Exception as e:
    print('using coordinates')
    x,y,z = [float(x) for x in argv[3].split(',')]
  
  r = float(argv[4])
  print('radius %g'%r)

  try:
    types = [int(x) for x in argv[5].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  rotation = None
  if len(argv) > 6:
    try:
      rotation = np.loadtxt(argv[6])
      print('rotation matrix:')
      with np.printoptions(precision=3, suppress=True):
        print(rotation)
    except:
      rotation = None
      print('no rotation')

  IDs = None
  if len(argv) > 7:
    try:
      IDs = np.fromfile(argv[7],dtype=np.uint32)
      print('using ID file')
    except:
      IDs = None
      print('no ID file')

  phys = False
  if len(argv) > 8:
    if argv[8][0].lower() == 't': phys = True
    elif argv[8][0].lower() == 'f': phys = False
    elif int(argv[8]) != 0: phys = True
  if phys:
    r_phys = r
    print('transform to physical')

  outbase = 'subfield'
  if len(argv) > 9:
    outbase = argv[9]
  
  names, headers = list_snapshots()

  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])

    if snapshot_number < ssmin or snapshot_number > ssmax:
      continue

    outname = outbase + filename[-4:] + '.bin'
    scale = headers[i]['Time']

    try:
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
    except:
      center = [x,y,z]

    if phys:
      r = r_phys / scale

    pos, mass, header = read_particles_filter(filename,center=center,halfwidth=r*1.1,rotation=rotation,type_list=types,ID_list=IDs,opts={'mass':True,'pos':True})

    pos += np.array([r,r,r]).reshape((1,3))

    if nnk is None:
      dens, bins = cic_bin(pos,2*r,GridSize,weights=mass,density=True)
    else:
      dens, bins = nn_density(pos,2*r,GridSize,weights=mass,nnk=nnk,simple=nns)

    if phys:
      dens /= scale**3

    dens.tofile(outname)
    print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
