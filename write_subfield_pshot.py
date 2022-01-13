import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter
from pathlib import Path
import h5py

# memory usage is 16 bytes per particle + 4 bytes per cell

# use loops for memory-efficiency
@njit
def ngp_bin(x,BoxSize,GridSize,weights,density):
  NP = x.shape[0]

  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)

  hist = np.zeros((N,N,N),dtype=np.float32)
  for p in range(NP):
    f = x[p] / dx # (3,)

    i = f.astype(np.int32)

    hist[i[0],i[1],i[2]] += weights[p]

  if density:
    hist /= dx**3

  return hist,bins

def group_subhalo_particle_count(fileprefix):
  filepath = [
    Path(fileprefix + '.hdf5'),
    Path(fileprefix + '.0.hdf5'),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d.hdf5'
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1

  fileinst = 0
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      numfiles = header['NumFiles']
      Nsubfile = header['Nsubgroups_ThisFile']
      if header['Ngroups_Total'] == 0:
        return None, None, None
      if fileinst == 0:
        Nsub = f['Group/GroupNsubs'][0]
        isub = 0
        npartGroup = np.array(f['Group/GroupLenType'][0])
        npartSub = np.array(f['Subhalo/SubhaloLenType'][0])
        npart = np.zeros_like(npartGroup)
      if isub + Nsubfile < Nsub:
        npart += np.sum(f['Subhalo/SubhaloLenType'],axis=0)
      else:
        npart += np.sum(f['Subhalo/SubhaloLenType'][:Nsub-isub],axis=0)
        break
      isub += Nsubfile

    fileinst += 1

  print(', '.join([str(x) for x in npartGroup]) + ' particles in group 0')
  print(', '.join([str(x) for x in npartSub]) + ' particles in subhalo 0')
  print(', '.join([str(x) for x in npart]) + ' particles in subhalos of group 0')

  return npartGroup, npartSub, npart

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot> <grid size> <x,y,z,r> [types=-1] [rotation-file=0] [group-file, if sub-only] [out-name]')
    return 1

  filename = argv[1]
  
  GridSize = int(argv[2])

  x,y,z,r = [float(x) for x in argv[3].split(',')]
  center = [x,y,z]
  w = 2. * r
  
  try:
    types = [int(x) for x in argv[4].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  try:
    rotation = np.loadtxt(argv[5])
    print('rotation matrix:')
    with np.printoptions(precision=3, suppress=True):
      print(rotation)
  except:
    rotation = None
    print('no rotation')

  try:
    groupname = argv[6]
    npartGroup, npartSub, npart = group_subhalo_particle_count(groupname)
    part_range = (npartSub,npart)
  except Exception as e:
    part_range = None

  try: outname = argv[7]
  except: outname = filename + '.field'
  
  # read
  pos, mass, header = read_particles_filter(filename,center=center,halfwidth=r,rotation=rotation,type_list=types,part_range=part_range,opts={'mass':True,'pos':True})
  print('%d particles'%len(mass) + ' in %g (Mpc/h)^3'%(w**3))

  pos += np.array([r,r,r]).reshape((1,3))

  pshot, bins = ngp_bin(pos,w,GridSize,weights=mass**2,density=True)

  pshot /= w**3 # make the dimension density^2 for consistency

  pshot.tofile(outname)
  print('saved P_shot / V_box grid to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
