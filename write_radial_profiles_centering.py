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

def find_center(pos,vel,mass,center,radius):
  particles = pos.shape[0]
  iterations = 0
  limit_particles = 100

  iok = slice(None)

  while particles > limit_particles:
    if iterations > 0:
      #iok = r2 < 4*radius**2 # just to save time
      radius *= 0.975

    r2 = np.sum((pos[iok] - center[None])**2,axis=1)
    ix = r2 < radius**2

    center = np.average(pos[iok][ix],axis=0,weights=mass[iok][ix])
    
    iterations += 1
    particles = np.sum(ix)
    print("  %d: CM=(%g %g %g), r=%g, N=%d"%(iterations,center[0],center[1],center[2],radius,particles))

  cv = np.average(vel[iok][ix],axis=0,weights=mass[iok][ix])
  return center, cv
  
def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot> <x,y,z,r,type> <rmin,rmax,nr> [types=-1] [out-name]')
    return 1

  filename = argv[1]

  center = np.array([float(x) for x in argv[2].split(',')[:3]])
  print('center (%g,%g,%g)'%tuple(center))
  try:
    radius = float(argv[2].split(',')[3])
    print('  search in radius %g'%radius)
    try:
      centype = int(argv[2].split(',')[4])
      print('  using type %d'%centype)
    except:
      centype = None
  except:
    radius = 0.
    centype = None
  
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

  try: outname = argv[5]
  except: outname = 'profiles' + ('' if centype is None else '%d'%centype) + filename[-4:] + '.txt'

  if centype is not None and radius > 0.:
    pos, vel, mass, header = read_particles_filter(filename,center=center,radius=rmax,type_list=[centype],opts={'mass':True,'pos':True,'vel':True})

  pos, vel, mass, typ, header = read_particles_filter(filename,opts={'mass':True,'pos':True,'vel':True,'type':True})

  if radius > 0.:
    icen = slice(None)
    if centype is not None:
      icen = centype == typ
    cx, cv = find_center(pos[icen],vel[icen],mass[icen],center,radius)
  else:
    cx = center
    cv = np.average(vel,axis=0,weights=mass)
  
  iprof = slice(None)
  if types is not None:
    iprof = np.isin(typ,types)

  r, rho, v2, vr2, vr, rv, ru, m, ct = profile(pos[iprof]-cx[None],vel[iprof]-cv[None],mass[iprof],rmin,rmax,nr)

  with open(outname,'wt') as f:
    f.write('# (%.6e, %.6e, %.6e)\n'%tuple(cx))
    f.write('# (%.6e, %.6e, %.6e)\n'%tuple(cv))
    f.write('# %.6e\n'%header['Time'])
    f.write('# radius rho r_upper mass count <v^2> <vr^2> <vr> <Lx> <Ly> <Lz>\n')
    for i in range(len(r)):
      f.write('%.6e %.6e %.6e %.6e %d %.6e %.6e %.6e %.6e %.6e %.6e\n'%(r[i], rho[i], ru[i], m[i], ct[i], v2[i], vr2[i], vr[i], rv[i,0], rv[i,1], rv[i,2]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
