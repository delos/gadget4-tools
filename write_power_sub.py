import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter
import h5py
from scipy.fft import fftn
from scipy.signal.windows import hann
from pathlib import Path

# memory usage is 16 bytes per particle + 4 bytes per cell

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

    for d in range(3):
      if f[d] < 0.5: f[d] += N
      if f[d] >= N+0.5: f[d] -= N

    i = (f-0.5).astype(np.int32)
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

@njit
def window_3D(delta,win):
  GridSize = len(win)
  for i in range(GridSize):
    for j in range(GridSize):
      for k in range(GridSize):
        delta[i,j,k] *= win[i]*win[j]*win[k]

@njit
def sinc(x):
  if x > 0.1:
    return np.sin(x)/x
  else:
    return 1. - x**2/6. + x**4/120. - x**6/5040. + x**8/362880.

@njit
def power_1d(deltak,BoxSize,deconvolve='CIC'):
 
  GridSize = deltak.shape[0]
  dx = BoxSize / GridSize
  dk = 2*np.pi/BoxSize
  kNfac = np.pi / GridSize
 
  # radial bins for k
  # corner of cube is at distance np.sqrt(3)/2*length from center
  k = np.zeros(GridSize//2)
  pk = np.zeros(GridSize//2)
  ct = np.zeros(GridSize//2,dtype=np.int32)
 
  for ix in range(GridSize):

    if ix < GridSize//2: kx = ix
    else: kx = ix-GridSize

    for iy in range(GridSize):

      if iy < GridSize//2: ky = iy
      else: ky = iy-GridSize

      for iz in range(GridSize):

        if iz < GridSize//2: kz = iz
        else: kz = iz-GridSize

        kf = np.sqrt(kx**2+ky**2+kz**2)

        k_ = kf * dk

        i = int(np.floor(kf)-1)
        if i < len(k) and i >= 0:
          pk_ = np.abs(deltak[ix,iy,iz])**2*(BoxSize/GridSize**2)**3

          if deconvolve == 'CIC':
            # deconvolve CIC kernel
            W = sinc(kx*kNfac) * sinc(ky*kNfac) * sinc(kz*kNfac)
            pk_ /= W**4

          # bin
          k[i] += np.log(k_)
          pk[i] += pk_
          ct[i] += 1

  k[ct>0] /= ct[ct>0]
  k = np.exp(k)
  pk[ct>0] /= ct[ct>0]

  return k, pk

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
    print('python script.py <snapshot> <grid size> <x,y,z,w> [types=-1] [rotation-file=0] [group-file, if sub-only] [out-name]')
    return 1

  filename = argv[1]
  
  GridSize = int(argv[2])

  x,y,z,w = [float(x) for x in argv[3].split(',')]
  center = [x,y,z]
  r = 0.5*w
  
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
    print(e)
    part_range = None

  try: outname = argv[7]
  except: outname = filename + '.power.txt'
  
  # read
  pos, mass, header = read_particles_filter(filename,center=center,halfwidth=r,rotation=rotation,type_list=types,part_range=part_range,opts={'mass':True,'pos':True})
  print('%d particles'%len(mass))

  pos += np.array([r,r,r]).reshape((1,3))

  Mtot = np.sum(mass)
  M2tot = np.sum(mass**2)
  Neff = Mtot**2/M2tot
  print('N_eff = %.0f'%Neff)
  pk_shot = w**3 / Neff
  density_mean = Mtot / w**3

  dens, bins = cic_bin(pos,w,GridSize,weights=mass,density=True)

  densmean = np.mean(dens)
  dens /= densmean
  dens -= 1
  print('density field constructed')
  print('mean density = %g = %g'%(density_mean,densmean))

  # window
  win = hann(GridSize)
  win /= np.sqrt(np.mean(win**2))
  window_3D(dens,win)
  print('windowed')

  # Fourier transform
  densk = fftn(dens,overwrite_x=True)
  print('FFT done')

  k,pk = power_1d(densk,w)
  #k,pk = power_1d(densk,w,deconvolve=None)

  np.savetxt(outname,np.stack((k,pk)).T,header='%g\n%g'%(pk_shot,density_mean))
  print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
