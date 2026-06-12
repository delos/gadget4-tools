import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter, read_subhalos
from scipy.spatial import KDTree
import multiprocessing
from multiprocessing.dummy import Pool
import os

@njit(nogil=True)
def mvir(X,x,m,rhovir,box):
  NP = x.shape[0]
  lnrmin, lnrmax, nr = np.log(1e-7*box),np.log(1e-1*box),1389
  mass = np.zeros(nr)
  for p in range(NP):
    dx = x[p] - X
    for d in range(3):
      if dx[d] <-0.5*box:
        dx[d] += box
      elif dx[d] > 0.5*box:
        dx[d] -= box
      #if np.abs(dx[d]) > 0.1*box:
      #  continue
    lnr = 0.5*np.log(dx[0]**2+dx[1]**2+dx[2]**2)
    f = (lnr - lnrmin)/(lnrmax-lnrmin) * nr
    if f >= nr:
      continue
    if f < 0.:
      i = 0
    else:
      i = int(f)
    mass[i] += m[p]
  mass = np.cumsum(mass)
  radius = np.exp(np.linspace(lnrmin,lnrmax,nr+1)[1:])
  density = mass / (4*np.pi/3*radius**3)
  ivir = np.where(density > rhovir)[0]
  if len(ivir) > 0:
    i0 = ivir[-1]
    f = (rhovir - density[i0])/(density[i0+1] - density[i0])
    return mass[i0] + f*(mass[i0+1]-mass[i0]), radius[i0] + f*(radius[i0+1]-radius[i0])
  return 0.,0.

def weigh(i,x,m,X,M,R,rhovir,box,tree):
  if i%100==99:
    print('  %d/%d'%(i+1,M.size))
  ix = tree.query_ball_point(X[i], 0.1*box,eps=0.1)
  M[i], R[i] = mvir(X[i],x[ix],m[ix],rhovir,box)

@njit(nogil=True)
def hostvir(i,M,R,X,sub,host,box):
  if M[i] > 0:
    dist = X[i+1:] - X[i:i+1]
    dist[dist>0.5*box] -= box
    dist[dist<-0.5*box] += box
    inside = np.sum(dist**2,axis=1) < R[i+1:]**2
    if np.any(inside):
      host[i] = sub[np.where(inside)[0][-1]]

def prune(i,M,R,X,sub,host,box):
  if i%100==99:
    print('  %d/%d'%(i+1,M.size))
  hostvir(i,M,R,X,sub,host,box)

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot-file> <group-file> [virial-density=200] [out-name]')
    return 1
  
  try:
    vir = float(argv[3])
  except:
    vir = 200.

  try:
    outname = argv[4]
    if outname[-4:] != '.npz':
      outname += '.npz'
  except:
    outname = argv[1] + '_subvir%.0f.npz'%vir

  # read particles
  pos, mass, header = read_particles_filter(argv[1],opts={'mass':True,'pos':True},reduce=None)
  box = header['BoxSize']
  rhomean = np.sum(mass) / box**3

  # read subhalos
  hpos, header = read_subhalos(argv[2],opts={'pos':True,},)
  NH = hpos.shape[0]
  M = np.zeros(NH,dtype=np.float32)
  R = np.zeros(NH,dtype=np.float32)
  sub = np.arange(NH)

  tree = KDTree(pos,boxsize=box)

  pool_size = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
  print('using %d CPUs'%pool_size)

  print('getting M_%.0f and R_%.0f'%(vir,vir))
  with Pool(pool_size) as p:
    p.map(lambda i: weigh(i,pos,mass,hpos,M,R,vir*rhomean,box,tree),np.arange(NH))
  '''
  for i in range(NH):
    if i%100 == 99 or True:
      print('  %d/%d'%(i+1,NH))
    weigh(i,pos,mass,hpos,M,R,vir*rhomean,box)
    #ix = np.all((pos-hpos[i:i+1] + 0.5*box)%box - 0.5*box < 0.1*box,axis=1)
    #r, m = np.sum(((pos[ix] - hpos[i:i+1] + 0.5*box)%box - 0.5*box)**2,axis=1)**0.5, mass[ix]
    #ix = r < 0.1*box
    #r, m = r[ix], m[ix]
    #sort = np.argsort(r)
    #r, m = r[sort], m[sort]
    #m = np.cumsum(m) - 0.5*m
    #rho = m / (4*np.pi/3*r**3)
    #ivir = np.where(rho > vir*rhomean)[0]
    #if len(ivir) > 0:
    #  i0 = ivir[-1]
    #  f = (vir*rhomean - rho[i0])/(rho[i0+1] - rho[i0])
    #  M[i] = m[i0] + f*(m[i0+1]-m[i0])
    #  R[i] = r[i0] + f*(r[i0+1]-r[i0])
  '''
  print('%d halos (of %d initial); now prune subhalos'%(np.sum(M>0.),NH))
  sort = np.argsort(M)
  X = hpos[sort]
  M = M[sort]
  R = R[sort]
  sub = sub[sort]
  host = np.full(NH,-1)
  with Pool(pool_size) as p:
    p.map(lambda i: prune(i,M,R,X,sub,host,box),np.arange(NH-1))
  M[host>=0] = 0.
  R[host>=0] = 0.
  '''
  for i in range(NH-1):
    if i%100 == 99:
      print('  %d/%d'%(i+1,NH))
    prune(i,M,R,X,sub,host,box)
  '''
  sort = np.argsort(M)[::-1]
  print('%d halos (of %d initial); now saving %s'%(np.sum(M>0.),NH,outname))
  np.savez(outname,X=X[sort],M=M[sort],R=R[sort],sub=sub[sort],host=host[sort],vir=vir,)

if __name__ == '__main__':
  from sys import argv
  run(argv)
