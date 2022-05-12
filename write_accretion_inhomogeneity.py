import numpy as np
from scipy.interpolate import interp1d
from numba import njit
from snapshot_functions import list_snapshots, read_particles_filter, group_extent
from scipy.spatial import cKDTree
#import sklearn

fileprefix_subhalo = 'groups_%03d/fof_subhalo_tab_%03d'
fileprefix_snapshot = 'snapdir_%03d/snapshot_%03d'

def run(argv):
  
  if len(argv) < 5:
    print('python script.py <snapshot min,max> <trace-file> <time-file> <NN k> [factor=1.17] [out-file]')
    return 1

  try:
    ssmin,ssmax = [int(x) for x in argv[1].split(',')]
  except:
    ssmin = ssmax = int(argv[1])

  _data = np.loadtxt(argv[2])
  _ss = _data[:,0]
  _x = _data[:,2]
  _y = _data[:,3]
  _z = _data[:,4]
  _grp = _data[:,10]
  print('using trace file %s'%argv[2])

  __data = np.loadtxt(argv[3])
  __ss = __data[:,0]
  __a = __data[:,1]
  print('using time file %s'%argv[3])

  nnk = int(argv[4])
  print('%d nearest neighbors'%nnk)
  
  try: fac = float(argv[5])
  except: fac = 1.17008878749642195041
  print('factor = %f'%fac)
  
  try: outfile = argv[6]
  except: outfile = 'accretion_inhomogeneity.txt'

  __ss_prev = interp1d(np.log(__a),__ss,bounds_error=False,fill_value=np.nan)(np.log(__a/fac))

  names, headers = list_snapshots()
  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])
    ssprev = __ss_prev[__ss==snapshot_number][0]
    ss0 = np.floor(ssprev).astype(np.int32)
    ss1 = np.ceil(ssprev).astype(np.int32)
    f = ssprev - ss0

    if ss0 < _ss[0] or snapshot_number < ssmin or snapshot_number > min(ssmax,_ss[-1]):
      continue

    print('snapshot = %d, previous snapshot = %.3f (%d, %d, %.3f)'%(snapshot_number,ssprev,ss0,ss1,f))

    groupfile = fileprefix_subhalo%(snapshot_number,snapshot_number)
    group = int(_grp[_ss==snapshot_number][0])

    scale = __a[__ss==snapshot_number][0]
    
    gpos, grad, header = group_extent(groupfile,group,size_definition='Mean200')

    IDs, header = read_particles_filter(filename,center=gpos,radius=grad,opts={'ID':True,})
    print('%d particles in halo'%len(IDs))

    agg_mass = []
    agg_rho = []
    agg_vol = []
    for ss in [ss0,ss1]:
      grpf = fileprefix_subhalo%(ss,ss)
      grp = int(_grp[_ss==ss][0])
      gp, gr, header = group_extent(grpf,grp,size_definition='Mean200')

      ssfile = fileprefix_snapshot%(ss,ss)
      pos, mass, header = read_particles_filter(ssfile,center=gp,radius=(gr,None),ID_list=IDs,
        opts={'mass':True,'pos':True},chunksize=2147483648)

      N = mass.size
      print('building tree (%d particles)'%N)
      tree = cKDTree(pos)
      print('querying nearest neighbors')

      rho = np.zeros_like(mass)

      chunksize = 1024**2
      istart = 0
      thresh = 0.
      while True:
        iend = min(istart+chunksize,N)
        dist,index = tree.query(pos[istart:iend],nnk)
        #tree = sklearn.neighbors.KDTree(pos[istart:iend])
        #dist,index = tree.query(pos,nnk)
        rho[istart:iend] = np.sum(mass[index],axis=-1)/(4./3*np.pi*dist[:,-1]**3)
        if iend >= N:
          break
        print('  %d/%d'%(iend,N))
        istart = iend

      
      agg_mass += [np.sum(mass)]
      agg_rho += [np.sum(rho*mass)/np.sum(mass)]
      agg_vol += [np.sum(mass/rho)]

    print(snapshot_number,scale,f,agg_mass[0],agg_mass[1],agg_vol[0],agg_vol[1],agg_rho[0],agg_rho[1])

    with open(outfile,'at') as fp:
      fp.write('%d %.6e %.6f %.6e %.6e %.6e %.6e %.6e %.6e\n'%(
        snapshot_number,scale,f,agg_mass[0],agg_mass[1],agg_vol[0],agg_vol[1],agg_rho[0],agg_rho[1]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
