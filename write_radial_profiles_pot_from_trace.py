import numpy as np
from numba import njit
from snapshot_functions import list_snapshots, read_particles_filter

@njit
def profile(pos,vel,mass,pot,rmin,rmax,nr):

  bin_ct = np.zeros(nr,dtype=np.int32)
  bin_m = np.zeros(nr)
  bin_mvr = np.zeros(nr)
  bin_mvr2 = np.zeros(nr)
  bin_mv2 = np.zeros(nr)
  bin_mrv = np.zeros((nr,3))
  bin_mpot = np.zeros(nr)
  low_ct = 0
  low_m = 0.

  logrmaxrmin = np.log(rmax/rmin)

  for p in range(pos.shape[0]):
    r = np.sqrt(np.sum(pos[p]**2))
    vr = np.sum(pos[p]*vel[p])/r
    v2 = np.sum(vel[p]**2)

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
    bin_mvr[i] += mass[p] * vr
    bin_mvr2[i] += mass[p] * vr**2
    bin_mv2[i] += mass[p] * v2
    bin_mpot[i] += mass[p] * pot[p]
    for d in range(3):
      bin_mrv[i,d] += mass[p] * (pos[p,(d+1)%3]*vel[p,(d+2)%3] - pos[p,(d+2)%3]*vel[p,(d+1)%3])

  bin_rl = rmin * (rmax/rmin)**(np.arange(nr)*1./nr) # lower edges
  bin_ru = rmin * (rmax/rmin)**(np.arange(1,nr+1)*1./nr) # upper edges
  bin_r = (0.5*(bin_rl**3 + bin_ru**3))**(1./3) # volume-averaged radius

  bin_vol = 4./3 * np.pi * (bin_ru**3-bin_rl**3) # bin volumes

  bin_rho = bin_m / bin_vol

  bin_mcum = np.cumsum(bin_m) + low_m

  bin_ctcum = np.cumsum(bin_ct) + low_ct

  bin_v2 = bin_mv2 / bin_m
  bin_vr2 = bin_mvr2 / bin_m
  bin_vr = bin_mvr / bin_m

  bin_rv = bin_mrv / bin_m.reshape((-1,1))

  bin_pot = bin_mpot / bin_m

  return bin_r, bin_rho, bin_v2, bin_vr2, bin_vr, bin_rv, bin_ru, bin_mcum, bin_ctcum, bin_pot

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot min,max> <trace-file> <rmin,rmax,nr> [types=-1] [out-prefix]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]

  _data = np.loadtxt(argv[2])
  if len(_data.shape) == 1:
    _data = _data[None,:]
  _ss = _data[:,0]
  _x  = _data[:,2]
  _y  = _data[:,3]
  _z  = _data[:,4]
  _vx = _data[:,5]
  _vy = _data[:,6]
  _vz = _data[:,7]
  print('using trace file')
  
  rb = argv[3].split(',')
  rmin = float(rb[0])
  rmax = float(rb[1])
  nr = int(rb[2])
  print('%d radial bins from %g to %g'%(nr,rmin,rmax))

  try:
    types = [int(x) for x in argv[4].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  
  try: outbase = argv[5]
  except: outbase = 'profiles'

  names, headers = list_snapshots()
  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])

    if snapshot_number < ssmin or snapshot_number > ssmax:
      continue

    outname = outbase + filename[-4:] + '.txt'

    if snapshot_number < _ss[0]:
      center = [_x[0],_y[0],_z[0]]
      center_v = np.zeros(3)
    elif snapshot_number > _ss[-1]:
      center = [_x[-1],_y[-1],_z[-1]]
      center_v = np.zeros(3)
    else:
      center = [
        _x[_ss==snapshot_number][0],
        _y[_ss==snapshot_number][0],
        _z[_ss==snapshot_number][0],
        ]
      center_v = np.array([
        _vx[_ss==snapshot_number][0],
        _vy[_ss==snapshot_number][0],
        _vz[_ss==snapshot_number][0],
        ])

    pos, vel, mass, pot, header = read_particles_filter(filename,center=center,radius=rmax,type_list=types,opts={'mass':True,'pos':True,'vel':True,'pot':True})

    vel -= center_v.reshape((1,3))

    r, rho, v2, vr2, vr, rv, ru, m, ct, phi = profile(pos,vel,mass,pot,rmin,rmax,nr)

    with open(outname,'wt') as f:
      f.write('# (%.12e, %.12e, %.12e)\n'%tuple(center))
      f.write('# %.12e\n'%header['Time'])
      f.write('# radius rho r_upper mass count <v^2> <vr^2> <vr> <Lx> <Ly> <Lz> Phi\n')
      for i in range(len(r)):
        f.write('%.6e %.6e %.6e %.6e %d %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n'%(r[i], rho[i], ru[i], m[i], ct[i], v2[i], vr2[i], vr[i], rv[i,0], rv[i,1], rv[i,2], phi[i]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
