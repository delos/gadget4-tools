import numpy as np
import os
from numba import njit
from snapshot_functions import group_data, read_particles_filter, read_params

fileprefix_snapshot = 'snapdir_%03d/snapshot_%03d'
fileprefix_subhalo = 'groups_%03d/fof_subhalo_tab_%03d'

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

    i = np.floor(f-0.5).astype(np.int32)
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

def run(argv):
  
  if len(argv) < 5:
    print('python script.py <snapshot min,max[,load-file]> <grid-file1,file2,...> <trace-file> <r_phys min,max> [types=-1] [r fac=2] [out-prefix]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')[:2]]
  try:
    loadfile = argv[1].split(',')[2]
  except:
    loadfile = None

  gridfiles = argv[2].split(',')
  GridSize = int((os.stat(gridfiles[0]).st_size//4)**(1./3)+.5)
  print('%d^3 cells in full grid'%(GridSize))

  _data = np.loadtxt(argv[3])
  _ss = _data[:,0]
  _a = _data[:,1]
  _grp = _data[:,10]
  pid = _data[0,12]
  print('using trace file %s'%argv[3])

  rmin_phys,rmax_phys = [float(x) for x in argv[4].split(',')]
  print('physical radii (%.5e,%.5e)'%(rmin_phys,rmax_phys))

  try:
    types = [int(x) for x in argv[5].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  if len(argv) > 6:
    rfac = float(argv[6])
  else:
    rfac = 2.
  print('radius factor %f'%rfac)

  outprefix = 'origin_%.3e_%.3e'%(rmin_phys,rmax_phys)
  if len(argv) > 7:
    outprefix = argv[7]
  print('output prefix: %s'%outprefix)
  outbase = outprefix + '_%d'

  ss0file = fileprefix_snapshot%(0,0)
  prefile = ss0file if loadfile is None else loadfile

  params = read_params(ss0file)
  if params['GravityConstantInternal'] > 0:
    G = params['GravityConstantInternal']
  else:
    G = 6.6743e-8 / params['UnitVelocity_in_cm_per_s']**2 * params['UnitMass_in_g'] / params['UnitLength_in_cm']
  rho_mean = 3 * (params['Hubble'] * params['HubbleParam'])**2/(8*np.pi*G) * params['Omega0']
  print('rho_mean = %e'%rho_mean)
  BoxSize = params['BoxSize']

  pos0, _ = read_particles_filter(prefile,ID_list=[pid],opts={'pos':True})
  pos0.shape = (3,)
  print('Lagrangian center = (%.6e,%.6e,%.6e)'%tuple(pos0))

  for ss in range(ssmin,ssmax+1):
    if ss < _ss[0]: continue
    if ss > _ss[-1]: break

    print('snapshot %d'%ss)
    ssfile = fileprefix_snapshot%(ss,ss)
    grpfile = fileprefix_subhalo%(ss,ss)
    grp = int(_grp[_ss==ss][0])
    a = _a[_ss==ss]
    rmin = rmin_phys/a
    rmax = rmax_phys/a

    gpos, grad, gmass, _ = group_data(grpfile,grp)
    lrad = (gmass / (4./3*np.pi*rho_mean))**(1./3)
    print('  group position = (%.6e,%.6e,%.6e)'%tuple(gpos) + ', radius = %.6e'%grad)
    print('  group mass = %.6e -> Lagrangian radius ~ %.6e'%(gmass,lrad))

    # get positions centered on group
    IDs, _ = read_particles_filter(ssfile,center=gpos,radius=(rmin,rmax),type_list=types,opts={'ID':True})
    print('  %d particles match'%(IDs.size))

    rad = lrad * rfac
    pos, mass, _ = read_particles_filter(prefile,center=pos0,radius=rad*2.,type_list=types,ID_list=IDs,opts={'mass':True,'pos':True},chunksize=2*1024**3)
    pos += pos0.reshape((1,3))

    cm = np.sum(pos*mass[:,None],axis=0)/np.sum(mass)
    
    string = '%d  %.7f  %.7f %.7f %.7f\n'%(ss,a,cm[0]/BoxSize,cm[1]/BoxSize,cm[2]/BoxSize)
    with open('origin_CM.txt','at') as fp:
      fp.write(string)
    print(string)

    cmGrid = (cm / BoxSize * GridSize).astype(int)
    cm = (cmGrid+.5) * BoxSize / GridSize

    radGrid = int(rad / BoxSize * GridSize + .5)
    rad = (radGrid+.5) * BoxSize / GridSize

    pos -= cm.reshape((1,3))
    pos += np.array([rad,rad,rad]).reshape((1,3))

    dens, bins = cic_bin(pos,2*rad,2*radGrid+1,weights=mass,density=True)
    dens /= rho_mean

    outname = outbase%ss
    dens.tofile(outname)
    print('  saved to ' + outname)

    xlim = np.array([cmGrid[0]-radGrid,cmGrid[0]+radGrid])
    ylim = np.array([cmGrid[1]-radGrid,cmGrid[1]+radGrid])
    zlim = np.array([cmGrid[2]-radGrid,cmGrid[2]+radGrid])

    for filename in gridfiles:
      with open(filename,'rb') as f:
        f.seek(GridSize**2*xlim[0]*4)
        dens = np.fromfile(f,count=GridSize**2*(xlim[1]-xlim[0]+1),dtype=np.float32)
      dens.shape = (xlim[1]-xlim[0]+1,GridSize,GridSize)
      dens = dens[:,ylim[0]:ylim[1]+1,zlim[0]:zlim[1]+1]
      dens.tofile(outname + '-' + os.path.basename(filename))

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

