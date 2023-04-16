import numpy as np
from numba import njit
from snapshot_functions import list_snapshots, read_header, read_particles_filter

@njit
def profile(pos,vel,mass,rmin,rmax,nr):

  bin_ct = np.zeros(nr,dtype=np.int32)
  bin_m = np.zeros(nr)
  bin_mvr = np.zeros(nr)
  bin_mvr2 = np.zeros(nr)
  bin_mv2 = np.zeros(nr)
  bin_mrv = np.zeros((nr,3))
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

  return bin_r, bin_rho, bin_v2, bin_vr2, bin_vr, bin_rv, bin_ru, bin_mcum, bin_ctcum

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot min,max> <trace-file> <rmin,rmax,nr> [types=-1] [out-prefix]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]

  _data = np.loadtxt(argv[2])
  _ss = _data[:,0]
  _t  = _data[:,1]
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

    header = read_header(filename)
    time = header['Time']

    if time < _t[0]:
      center = [_x[0],_y[0],_z[0]]
      center_v = np.zeros(3)
    elif time > _t[-1]:
      center = [_x[-1],_y[-1],_z[-1]]
      center_v = np.zeros(3)
    else:
      center = [
        np.interp(time,_t,_x),
        np.interp(time,_t,_y),
        np.interp(time,_t,_z),
        ]
      center_v = np.array([
        np.interp(time,_t,_vx),
        np.interp(time,_t,_vy),
        np.interp(time,_t,_vz),
        ])

    pos, vel, mass, header = read_particles_filter(filename,center=center,radius=rmax,type_list=types,opts={'mass':True,'pos':True,'vel':True})

    vel -= center_v.reshape((1,3))

    r, rho, v2, vr2, vr, rv, ru, m, ct = profile(pos,vel,mass,rmin,rmax,nr)

    with open(outname,'wt') as f:
      f.write('# (%.12e, %.12e, %.12e)\n'%tuple(center))
      f.write('# %.12e\n'%header['Time'])
      f.write('# radius rho r_upper mass count <v^2> <vr^2> <vr> <Lx> <Ly> <Lz>\n')
      for i in range(len(r)):
        f.write('%.6e %.6e %.6e %.6e %d %.6e %.6e %.6e %.6e %.6e %.6e\n'%(r[i], rho[i], ru[i], m[i], ct[i], v2[i], vr2[i], vr[i], rv[i,0], rv[i,1], rv[i,2]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
