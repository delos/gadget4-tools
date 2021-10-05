import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter
import h5py
from scipy.fft import fftn
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
    #if f[0] < 0.5 or f[1] < 0.5 or f[2] < 0.5 or f[0] >= N+0.5 or f[1] >= N+0.5 or f[2] >= N+0.5:
    #  continue

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

@njit
def window(x):
  if x > .5 or x < 0:
    return 0
  return np.cos(np.pi*x)**2
window_mean = 0.102644 # over cube, not sphere
window_rms = 0.217122

@njit
def apply_window(delta):
  GridSize = delta.shape[0]
  for i in range(GridSize):
    x = (i + 0.5) / GridSize - 0.5
    for j in range(GridSize):
      y = (j + 0.5) / GridSize - 0.5
      for k in range(GridSize):
        z = (k + 0.5) / GridSize - 0.5
        r = np.sqrt(x**2+y**2+z**2)
        delta[i,j,k] *= window(r) / window_rms

@njit
def get_weights(x,BoxSize):
  NP = x.shape[0]

  weights = np.zeros(NP)
  for p in range(NP):
    r = np.sqrt(np.sum(x[p]**2))
    weights[p] = window(r/BoxSize)
  return weights

@njit
def einasto(r,rs,ps,a):
  return ps * np.exp(-2./a*((r/rs)**a-1))

@njit
def subtract_profile(field,BoxSize,pos,profile,args):
  avg = 0.
  avgr = 0.
  GridSize = field.shape[0]
  for i in range(GridSize):
    x = ((i + 0.5) / GridSize - 0.5) * BoxSize + pos[0]
    for j in range(GridSize):
      y = ((j + 0.5) / GridSize - 0.5) * BoxSize + pos[1]
      for k in range(GridSize):
        z = ((k + 0.5) / GridSize - 0.5) * BoxSize + pos[2]
        
        r = np.sqrt(x**2+y**2+z**2)

        field[i,j,k] -= profile(r,*args)
        avg += profile(r,*args)
        avgr += r
  print(avg / GridSize**3)
  print(avgr / GridSize**3)

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot> <grid size> <center x,y,z> <offset x,y,z,r> <profile rho_-2,r_-2,alpha> [types=-1] [group-file, if sub-only] [out-name]')
    return 1

  filename = argv[1]
  
  GridSize = int(argv[2])

  center = np.array([float(x) for x in argv[3].split(',')])

  x,y,z,r = [float(x) for x in argv[4].split(',')]
  offset = np.array([x,y,z])
  w = 2*r

  rhos,rs,alpha = [float(x) for x in argv[5].split(',')]
  
  try:
    types = [int(x) for x in argv[6].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  try:
    groupname = argv[7]
    npartGroup, npartSub, npart = group_subhalo_particle_count(groupname)
    part_range = (npartSub,npart)
  except Exception as e:
    part_range = None

  try: outname = argv[8]
  except: outname = filename + '.power.txt'
  
  # read
  print('reading particles in radius %g'%r+' centered at %g,%g,%g'%tuple(center+offset))
  pos, mass, header = read_particles_filter(filename,center=center+offset,radius=r,type_list=types,part_range=part_range,opts={'mass':True,'pos':True})
  print('%d particles'%len(mass) + ' in %g (Mpc/h)^3'%(4./3*np.pi*r**3))

  # get shot noise
  weights = get_weights(pos,w)
  pk_shot = np.sum((mass*weights)**2) / (w**3 * window_rms**2)
  density_mean = np.sum(mass*weights) / (w**3 * window_mean)

  # shift for density field
  pos += np.array([r,r,r]).reshape((1,3))

  dens, bins = cic_bin(pos,w,GridSize,weights=mass,density=True)
  print('density field constructed')

  subtract_profile(dens,w,offset,einasto,(rs,rhos,alpha))
  print('subtracted host profile')

  densmean = np.mean(dens)
  #dens -= densmean
  print('mean density = %g (residual %g)'%(density_mean,densmean))

  # window
  apply_window(dens)
  print('windowed')

  # Fourier transform
  densk = fftn(dens,overwrite_x=True)
  print('FFT done')

  k,pk = power_1d(densk,w)

  np.savetxt(outname,np.stack((k,pk)).T,header='%g\n%g'%(pk_shot,density_mean))
  print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)
