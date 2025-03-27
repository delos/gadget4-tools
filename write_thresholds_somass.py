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
    print('python script.py <mass-file> <snap-file> <out-file>')
    return 1

  rstr_list = []
  for i in range(6):
    Tlist = glob('T%d%d_*'%a_b_from_ab[i])
    rstr_list = np.unique(np.concatenate((rstr_list,[x[4:] for x in Tlist])))

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

  data = np.load(argv[1])
  ID = data['ID']
  out = {'D':D,'ID':ID,}
  for Mstr in ['Mmax','Mmin']:
    M = data[Mstr]
    irM = np.interp(np.log(M),np.log(M_list),np.arange(len(M_list),dtype=np.float32),left=np.nan,right=np.nan)
    irM0 = np.floor(irM).astype(np.int32)
    irM1 = irM0+1
    frM = irM-irM0
    for i in range(6):
      print('T_{%d%d}'%a_b_from_ab[i])
      T = np.zeros(len(M),dtype=np.float32)
      for ir,rstr in enumerate(rstr_list):
        j0 = irM0==ir
        j1 = irM1==ir
        if np.sum(j0) == 0 and np.sum(j1) == 0:
          continue
        file = 'T%d%d_%s'%(*a_b_from_ab[i],rstr)
        GridSize = int((os.stat(file).st_size//4)**(1./3)+.5)
        with open(file,'rb') as f:
          field = np.fromfile(f,count=GridSize**3,dtype=np.float32)
          #field.shape = (GridSize,GridSize,GridSize)
        #a_ic = ((ID-ID0)//GridSize**2)%GridSize
        #b_ic = ((ID-ID0)//GridSize)%GridSize
        #c_ic = (ID-ID0)%GridSize

        print('  reading %s (%d^3), %d+%d'%(file,GridSize,np.sum(j0),np.sum(j1)))

        if ir < len(rstr_list)-1:
          T[j0] += (1. - frM[j0]) * field[ID[j0]-ID0]
        if ir > 0:
          T[j1] += frM[j1] * field[ID[j1]-ID0]
      out['T%d%d_%s'%(*a_b_from_ab[i],Mstr)] = T
  np.savez(argv[3],**out)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

