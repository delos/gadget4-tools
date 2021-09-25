import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter

@njit
def profile(pos,mass,rmin,rmax,nr):

  bin_ct = np.zeros(nr,dtype=np.int32)
  bin_m = np.zeros(nr)
  low_ct = 0
  low_m = 0.

  logrmaxrmin = np.log(rmax/rmin)

  for p in range(mass.size):
    r = np.sqrt(np.sum(pos[p]**2))

    if r >= rmax:
      continue
    if r < rmin:
      low_ct += 1
      low_m += mass[p]
      continue

    i = int(np.log(r/rmin)/logrmaxrmin * nr)
    if i < 0 or i >= nr:
      raise Exception('...')

    bin_ct[i] += 1
    bin_m[i] += mass[p]

  bin_rl = rmin * (rmax/rmin)**(np.arange(nr)*1./nr)
  bin_ru = rmin * (rmax/rmin)**(np.arange(1,nr+1)*1./nr)
  bin_r = (0.5*(bin_rl**3 + bin_ru**3))**(1./3) # volume-averaged radius

  bin_vol = 4./3 * np.pi * (bin_ru**3-bin_rl**3)

  bin_rho = bin_m / bin_vol

  bin_mcum = np.cumsum(bin_m) + low_m

  bin_ctcum = np.cumsum(bin_ct) + low_ct

  return bin_r, bin_rho, bin_ru, bin_mcum, bin_ctcum

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot-file> <x,y,z> <rmin,rmax,nr> [types=-1] [output]')
    return 1
  
  c = np.array(argv[2].split(','),dtype=np.float64)

  rb = argv[3].split(',')
  rmin = float(rb[0])
  rmax = float(rb[1])
  nr = int(rb[2])
  
  try:
    types = [int(x) for x in argv[4].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  try: outname = argv[5]
  except: outname = argv[1] + '_rad.txt'

  # read particles and halos
  pos, mass, header = read_particles_filter(argv[1],center=c,radius=rmax,type_list=types,opts={'mass':True,'pos':True})

  r, rho, ru, m, ct = profile(pos,mass,rmin,rmax,nr)

  with open(outname,'wt') as f:
    f.write('# (%.12e, %.12e, %.12e)\n'%tuple(c))
    f.write('# %.12e\n'%header['Time'])
    f.write('# radius rho r_upper mass count\n')
    for i in range(len(r)):
      f.write('%.6e %.6e %.6e %.6e %d\n'%(r[i], rho[i], ru[i], m[i], ct[i]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
