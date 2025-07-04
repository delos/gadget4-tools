import numpy as np
import os
from snapshot_functions import read_header, read_params
from numba import njit
from glob import glob

# indexing
a_b_from_ab = [
  (0,0),
  (0,1),
  (0,2),
  (1,1),
  (1,2),
  (2,2),
  ]
ab_from_a_b = [
  [0,1,2],
  [1,3,4],
  [2,4,5],
  ]
ID0 = 1

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <mass-file> <snap-file> <out-file> [window=1] [N_min=500]')
    return 1
  window = int(argv[4]) if len(argv) > 4 else 1
  N0 = float(argv[5]) if len(argv) > 5 else 500.
  print(' '.join(argv))
  print('window=%d, N_min=%.0f'%(window,N0))

  rstr_list = []
  for i in range(6):
    Tlist = glob('T%d%d_%d_*'%(*a_b_from_ab[i],window))
    rstr_list = np.unique(np.concatenate((rstr_list,[x.split('_')[-1] for x in Tlist])))
  print(rstr_list)

  header = read_header(argv[2])
  box = header['BoxSize']
  Npar = header['NumPart_Total'][1]
  a = 1./(header['Redshift']+1.)
  mpar = header['MassTable'][1]

  density = mpar * Npar / box**3
  
  params = read_params(argv[2])
  D = header['Time']/params['TimeBegin']

  if params['GravityConstantInternal'] > 0:
    G = params['GravityConstantInternal']
  else:
    G = 6.6743e-8 / params['UnitVelocity_in_cm_per_s']**2 * params['UnitMass_in_g'] / params['UnitLength_in_cm']
  density_test = 3 * (params['Hubble'] * params['HubbleParam'])**2/(8*np.pi*G) * params['Omega0']
  print('a = %e [= %e]'%(header['Time'],a))
  print('density = %e [= %e]'%(density,density_test))

  r_list = np.array([float(s) for s in rstr_list]) * box
  M_list = 4./3 * np.pi * r_list**3 * density

  M0 = N0 * mpar

  data = np.load(argv[1])
  ID = data['ID']
  out = {'a':a,'D':D,'box':box,'rho':density,'m':mpar,}
  Mstrs = [l for l in list(data) if l[0]=='M']
  print('mass fields: ' + ','.join(Mstrs))
  for Mstr in Mstrs:
    M = data[Mstr]
    i0 = M > M0
    out['ID_%s'%Mstr] = ID[i0]
    out[Mstr] = M[i0]
    irM = np.interp(np.log(M[i0]),np.log(M_list),np.arange(len(M_list),dtype=np.float32),left=np.nan,right=np.nan)
    irM0 = np.floor(irM).astype(np.int32)
    irM1 = irM0+1
    frM = irM-irM0
    for i in range(6):
      print('T_{%d%d}'%a_b_from_ab[i])
      T = np.zeros(np.sum(i0),dtype=np.float32)
      for ir,rstr in enumerate(rstr_list):
        j0 = irM0==ir
        j1 = irM1==ir
        if np.sum(j0) == 0 and np.sum(j1) == 0:
          continue
        file = 'T%d%d_%d_%s'%(*a_b_from_ab[i],window,rstr)
        GridSize = int((os.stat(file).st_size//4)**(1./3)+.5)
        with open(file,'rb') as f:
          field = np.fromfile(f,count=GridSize**3,dtype=np.float32) * D
          #field.shape = (GridSize,GridSize,GridSize)
        #a_ic = ((ID-ID0)//GridSize**2)%GridSize
        #b_ic = ((ID-ID0)//GridSize)%GridSize
        #c_ic = (ID-ID0)%GridSize

        print('  reading %s (%d^3), %d+%d'%(file,GridSize,np.sum(j0),np.sum(j1)))

        if ir < len(rstr_list)-1:
          T[j0] += (1. - frM[j0]) * field[ID[i0][j0]-ID0]
        if ir > 0:
          T[j1] += frM[j1] * field[ID[i0][j1]-ID0]
      out['T%d%d_%s'%(*a_b_from_ab[i],Mstr)] = T
  np.savez(argv[3],**out)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

