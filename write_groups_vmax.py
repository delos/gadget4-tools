import numpy as np
from scipy.spatial import KDTree
from snapshot_functions import read_subhalos, read_params, read_particles_filter
from numba import njit

@njit
def sep_in_box(dx,box):
  '''
  put dx in range (-box/2, box/2)
  '''
  return (dx + 0.5*box)%box - 0.5*box

@njit
def group_particles(x,m,x0,r0,box):
  out_r = np.zeros(x.shape[0],dtype=np.float32)
  out_m = np.zeros(x.shape[0],dtype=np.float32)
  ii = 0
  for i in range(x.shape[0]):
    dx = sep_in_box(x[i] - x0,box)

    r2 = np.sum(dx**2)
    if r2 <= r0**2:
      out_r[ii] = np.sqrt(r2)
      out_m[ii] = m[i]
      ii += 1
  return out_r[:ii], out_m[:ii]

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <snapshot-file> <group-file> [output-file]')
    return 1
  
  try: outfile = argv[3]
  except: outfile = 'groupvmax.txt'

  params = read_params(argv[1])
  G = 6.6743e-8 / (params['UnitVelocity_in_cm_per_s']**2 * params['UnitLength_in_cm'] / params['UnitMass_in_g'])
  print('G = %.6e'%G)

  # read halos
  pos, radius, header = read_subhalos(argv[2],opts={},
    group_opts={'pos':True,'soradius':True},sodef='Mean200')
  NG = radius.size

  print('%d groups'%NG)

  rmax = np.zeros(NG,dtype=np.float32)
  vmax = np.zeros(NG,dtype=np.float32)

  x, m, header = read_particles_filter(argv[1],opts={'pos':True,'mass':True})

  tree = KDTree(x,boxsize=header['BoxSize'])
  print('tree done')

  for i in range(NG):
    #x, m, header = read_particles_filter(argv[1],center=pos[i],radius=radius[i],opts={'pos':True,'mass':True})

    #r = np.sqrt(np.sum(sep_in_box(x-pos[i:i+1],header['BoxSize'])**2,axis=1))

    #print('ix')
    #ix = r<radius[i]
    #r_ = r[ix]
    #m_ = m[ix]

    #r_, m_ = group_particles(x,m,pos[i],radius[i],header['BoxSize'])

    if radius[i] > 0.:
      indices = tree.query_ball_point(pos[i], radius[i], return_sorted=True)
      r_ = np.sqrt(np.sum(sep_in_box(x[indices]-pos[i:i+1],header['BoxSize'])**2,axis=1))
      m_ = m[indices]

      sort = np.argsort(r_)
      r_ = r_[sort]
      m_ = m_[sort]

      mcum = np.cumsum(m_)
      #vcirc = np.sqrt(G*mcum/(r_+params['SofteningComovingClass1']))
      vcirc = np.sqrt(G*mcum[:-1]/r_[1:])
      imax = np.argmax(vcirc)
      rmax[i] = r_[1:][imax]
      vmax[i] = vcirc[imax]

  if outfile.endswith('.npz'):
    np.savez(outfile,rmax=rmax,vmax=vmax,r200=radius,a=header['Time'])
  else:
    np.savetxt(outfile,np.concatenate((rmax[:,None],vmax[:,None],radius[:,None]),axis=1),header='%.7e'%header['Time'])

if __name__ == '__main__':
  from sys import argv
  run(argv)
