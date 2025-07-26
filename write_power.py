import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter
import h5py
from scipy.fft import fftn
from scipy.signal.windows import hann
from scipy.optimize import least_squares
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

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot> <grid size> [types=-1] [out-name]')
    return 1

  filename = argv[1]
  
  GridSize = int(argv[2])

  try:
    types = [int(x) for x in argv[3].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  try: outname = argv[4]
  except: outname = filename + '.power.txt'
  
  # read
  pos, mass, header = read_particles_filter(filename,type_list=types,opts={'mass':True,'pos':True})

  Mtot = np.sum(mass)
  M2tot = np.sum(mass**2)
  Neff = Mtot**2/M2tot
  print('N_eff = %.0f'%Neff)
  Veff = header['BoxSize']**3
  density_mean = Mtot / Veff
  pk_shot = Mtot**2 / (Neff*Veff)

  dens, bins = cic_bin(pos,header['BoxSize'],GridSize,weights=mass,density=True)

  densmean = np.mean(dens)
  dens -= densmean
  print('density field constructed')
  print('mean density = %g = %g'%(density_mean,densmean))

  # Fourier transform
  densk = fftn(dens,overwrite_x=True)
  print('FFT done')

  k,pk = power_1d(densk,header['BoxSize'])

  np.savetxt(outname,np.stack((k,pk)).T,header='%g\n%g'%(pk_shot,density_mean))
  print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)

