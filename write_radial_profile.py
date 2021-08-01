import numpy as np
from numba import njit
from snapshot_functions import gadget_to_particles

@njit
def center_box(c,pos,BoxSize):
  pos -= c.reshape((3,1))
  for p in range(pos.shape[1]):
    for d in range(3):
      if pos[d,p] > 0.5*BoxSize: pos[d,p] -= BoxSize
      if pos[d,p] < -0.5*BoxSize: pos[d,p] += BoxSize

@njit
def cm_in_radius(pos,mass,length,shift,radius,BoxSize):
  cm = np.zeros(3)
  m = 0.
  particles = 0
  idx = 0

  for p in range(length):
    dx = pos[:,p] - shift
    for d in range(3):
      if dx[d] >= 0.5*BoxSize: dx[d] -= BoxSize
      elif dx[d] < -0.5*BoxSize: dx[d] += BoxSize
    r2 = np.sum(dx**2)
    if r2 < radius**2:
      cm += pos[:,p] * mass[p]
      m += mass[p]
      particles += 1
    if r2 < 4*radius**2:
      for d in range(3):
        pos[d,idx], pos[d,p] = pos[d,p], pos[d,idx]
      mass[idx], mass[p] = mass[p], mass[idx]
      idx += 1

  cm /= m

  return cm, particles, idx
  
def find_center(c,pos,mass,radius,BoxSize):
  limit_particles = 100

  particles = pos.shape[1]
  if radius <= 0.: particles = 0

  shift = c.copy()

  iteration = 0
  idx = pos.shape[1]
  while particles > limit_particles:
    shift, particles, idx = cm_in_radius(pos,mass,idx,shift,radius,BoxSize)

    iteration += 1
    print('%4d: (%.12e, %.12e, %.12e), r=%6g, N=%d'%(iteration,shift[0],shift[1],shift[2],radius,particles))
    radius *= 0.975

  return shift

@njit
def profile(pos,mass,rmin,rmax,nr):

  bin_ct = np.zeros(nr,dtype=np.int32)
  bin_m = np.zeros(nr)
  low_ct = 0
  low_m = 0.

  logrmaxrmin = np.log(rmax/rmin)

  for p in range(pos.shape[1]):
    r = np.sqrt(np.sum(pos[:,p]**2))

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
    print('python script.py <snapshot-file> <x,y,z> <rmin,rmax,nr> [rfind=0] [output]')
    return 1
  
  c = np.array(argv[2].split(','),dtype=np.float64)

  rb = argv[3].split(',')
  rmin = float(rb[0])
  rmax = float(rb[1])
  nr = int(rb[2])
  
  try: rfind = float(argv[4])
  except: rfind = 0.

  try: outname = argv[5]
  except: outname = argv[1] + '_rad.txt'

  # read particles and halos
  pos, mass, header = gadget_to_particles(argv[1],opts={'pos':True,'mass':True})

  BoxSize = header['BoxSize']

  shift = find_center(c,pos,mass,rfind,BoxSize)
  center_box(shift,pos,BoxSize)

  r, rho, ru, m, ct = profile(pos,mass,rmin,rmax,nr)

  with open(outname,'wt') as f:
    f.write('# (%.12e, %.12e, %.12e)\n'%tuple(shift))
    f.write('# %.12e\n'%header['Time'])
    f.write('# radius rho r_upper mass count\n')
    for i in range(len(r)):
      f.write('%.6e %.6e %.6e %.6e %d\n'%(r[i], rho[i], ru[i], m[i], ct[i]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
