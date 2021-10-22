import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter, read_subhalos

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
    print('python script.py <snapshot-file> <group-file> <rmin,rmax,nr> [types=-1] [output-base] [start at #]')
    return 1
  
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

  try: outbase = argv[5]
  except: outbase = argv[1] + '_rad'

  try: start = int(argv[6])
  except: start = 0

  # read halos
  hpos, hlen, header = read_subhalos(argv[2],opts={'pos':True,'lentype':True})
  NH = hlen.shape[0]
  NT = hlen.shape[1]

  print('%d halos'%NH)

  part_index = np.zeros(NT,dtype=int)
  for i in range(start,NH):
    c = hpos[i]
    l = hlen[i]
      
    # read particles
    pos, mass, header = read_particles_filter(argv[1],center=c,part_range=(part_index,part_index+l),type_list=types,opts={'mass':True,'pos':True})
    part_index += l

    # get profile
    r, rho, ru, m, ct = profile(pos,mass,rmin,rmax,nr)

    # write profile
    outname = outbase + '_%d.txt'%i

    with open(outname,'wt') as f:
      f.write('# (%.12e, %.12e, %.12e)\n'%tuple(c))
      f.write('# %.12e\n'%header['Time'])
      f.write('# radius rho r_upper mass count\n')
      for i in range(len(r)):
        f.write('%.6e %.6e %.6e %.6e %d\n'%(r[i], rho[i], ru[i], m[i], ct[i]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
