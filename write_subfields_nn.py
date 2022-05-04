import numpy as np
from numba import njit
from snapshot_functions import list_snapshots, read_particles_filter
from pathlib import Path
import h5py
from scipy.spatial import cKDTree

def nn_density(x,BoxSize,GridSize,weights,nnk):
  NP = x.shape[0]

  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)

  hist = np.zeros((N,N,N),dtype=np.float32)

  tree = cKDTree(x / dx - 0.5)
  # particles in (-0.5,N-0.5)
  # put on grid 0,...,N-1

  for i in range(GridSize):
    if i%(GridSize//8) == 0:
      print('%d/%d'%(i,GridSize))
    grid = np.stack([np.full((GridSize,GridSize),i)]+np.meshgrid(*([np.arange(GridSize)]*2)),axis=-1)
    # grid has shape (GridSize,GridSize,3)
    dist,index = tree.query(grid,nnk)
    # dist, index have shape (GridSize,GridSize,nnk)
    hist[i] = nnk*(nnk+1.)/2. / (4./3*np.pi*np.sum(dist**3/weights[index],axis=-1)) # weighted M/V

  hist /= dx**3
  return hist,bins

def run(argv):
  
  if len(argv) < 5:
    print('python script.py <snapshot min,max> <grid size,k> <x,y,z or trace-file> <r> [types=-1] [rotation-file=0] [ID-file=0] [physical=0] [out-name]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]
  
  try:
    GridSize,nnk = [int(x) for x in argv[2].split(',')]
  except:
    GridSize = int(argv[2])
    nnk = 50

  print('%d^3 cells; k=%d nearest neighbors'%(GridSize,nnk))

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

  if len(argv) > 7:
    IDs = np.fromfile(argv[7],dtype=np.uint32)
  else:
    IDs = None

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

    dens, bins = nn_density(pos,2*r,GridSize,weights=mass,nnk=nnk)

    if phys:
      dens /= scale**3

    dens.tofile(outname)
    print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
