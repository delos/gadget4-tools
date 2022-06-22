import numpy as np
from numba import njit
from snapshot_functions import group_data, read_particles_filter, read_params

fileprefix_snapshot = 'snapdir_%03d/snapshot_%03d'
fileprefix_subhalo = 'groups_%03d/fof_subhalo_tab_%03d'

def run(argv):
  
  if len(argv) < 5:
    print('python script.py <snapshot min,max> <trace-file> <r_phys min,max> [types=-1] [outfile]')
    return 1

  try:
    ssmin,ssmax = [int(x) for x in argv[1].split(',')]
  except:
    ssmin = 0
    ssmax = 2**16

  _data = np.loadtxt(argv[2])
  _ss = _data[:,0]
  _a = _data[:,1]
  _grp = _data[:,10]
  pid = _data[0,12]
  print('using trace file %s'%argv[2])

  rmin_phys,rmax_phys = [float(x) for x in argv[3].split(',')]
  print('physical radii (%.5e,%.5e)'%(rmin_phys,rmax_phys))

  try:
    types = [int(x) for x in argv[4].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  outname = 'origin_CM.txt'
  if len(argv) > 5:
    outname = argv[5]

  ss0file = fileprefix_snapshot%(0,0)

  params = read_params(ss0file)
  if params['GravityConstantInternal'] > 0:
    G = params['GravityConstantInternal']
  else:
    G = 6.6743e-8 / params['UnitVelocity_in_cm_per_s']**2 * params['UnitMass_in_g'] / params['UnitLength_in_cm']
  rho_mean = 3 * (params['Hubble'] * params['HubbleParam'])**2/(8*np.pi*G) * params['Omega0']
  print('rho_mean = %e'%rho_mean)
  BoxSize = params['BoxSize']

  pos0, _ = read_particles_filter(ss0file,ID_list=[pid],opts={'pos':True})
  pos0.shape = (3,)
  print('Lagrangian center ~ (%.6e,%.6e,%.6e)'%tuple(pos0))

  lrad = 2

  with open(outname,'wt') as fp:

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

      # to save time, we only look for group particles within twice the group's Lagrangian radius
      rad = lrad * rfac
      pos, mass, _ = read_particles_filter(ss0file,center=pos0,radius=rad,type_list=types,ID_list=IDs,opts={'mass':True,'pos':True},chunksize=2*1024**3)

      cm = np.sum(pos*mass[:,None],axis=0)/np.sum(mass)

      cm /= BoxSize

      string = '%d  %.7f  %.7f %.7f %.7f\n'%(ss,a,cm[0],cm[1],cm[2])

      fp.write(string)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

