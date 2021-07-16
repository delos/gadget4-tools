import numpy as np
from scipy.fft import fftn
from numba import njit
import os
import matplotlib.pyplot as plt

@njit
def power_1d(delta,BoxSize):
  
  GridSize = delta.shape[0]
  dk = 2*np.pi/BoxSize
  
  nbins = int((GridSize+1) * np.sqrt(3)/2)

  hist_pk = np.zeros(nbins)
  hist_ct = np.zeros(nbins,dtype=np.int32)
  hist_k = np.zeros(nbins)
  
  # get wavenumbers associated with k-space grid
  for i in range(GridSize):
    kx = i
    if kx >= GridSize//2: kx -= GridSize
    for j in range(GridSize):
      ky = j
      if ky >= GridSize//2: ky -= GridSize
      for k in range(GridSize):
        kz = k
        if kz >= GridSize//2: kz -= GridSize

        k_mag = np.sqrt(kx**2+kz**2+kz**2)

        k_idx = int(k_mag)
        if k_idx >= nbins:
          continue
        
        hist_pk[k_idx] += np.abs(delta[i,j,k])**2*BoxSize**3/GridSize**6
        hist_ct[k_idx] += 1
        hist_k[k_idx] += k_mag*dk

  return hist_k[1:]/hist_ct[1:], hist_pk[1:]/hist_ct[1:]

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <box size>')
    return 1
  
  filename = argv[1]
  
  BoxSize = float(argv[2])

  dsize = os.stat(filename).st_size//4
  GridSize = int(dsize**(1./3)+.5)
  with open(filename,'rb') as f:
    delta = np.fromfile(f,count=-1,dtype=np.float32)
  delta.shape = (GridSize,GridSize,GridSize)

  delta = fftn(delta,overwrite_x=True)

  k,P = power_1d(delta,BoxSize)

  ax = plt.figure().gca()

  ax.loglog(k,k**3/(2*np.pi**2)*P)
  ax.set_xlim(k[0],k[-1])
  ax.set_xlabel(r'$k$ ($h$/kpc)')
  ax.set_ylabel(r'dimensionless $\mathcal{P}(k)$')

  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
