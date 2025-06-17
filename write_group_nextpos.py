import numpy as np
from numba import njit
from snapshot_functions import read_subhalos, fileprefix_subhalo, particles_by_ID, fileprefix_snapshot, read_particles_filter

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <ss #>')
    return 1
  
  ss0 = int(argv[1])

  ss = ss0
    
  sublen, length, sub1, header = read_subhalos(fileprefix_subhalo%(ss,ss),opts={'lentype':True},group_opts={'lentype':True,'firstsub':True},)
  sublen = sublen[sub1]
  bins = np.concatenate((np.zeros(6,dtype=int)[None],np.cumsum(length,axis=0)))
  subbins = np.concatenate((np.zeros(6,dtype=int)[None],np.cumsum(sublen,axis=0)))
  IDs = []
  for typ in range(6):
    ID, header = read_particles_filter(fileprefix_snapshot%(ss,ss),type_list=[typ],opts={'ID':True},part_range=(bins[0],bins[-1]))
    if typ == 1:
      box = header['BoxSize']
    ix = np.full(bins[-1,typ],False)
    for i in range(len(length[:,typ])):
      ix[bins[i,typ]:bins[i,typ]+sublen[i,typ]] = True
    IDs += [ID[ix]]

  ss = ss0 + 1

  for ii, ss in enumerate([ss0,ss0+1]):
    Xa, Xb, Xc, X, V = [np.zeros((np.shape(length)[0],3)) for i in range(5)]
    M = np.zeros(np.shape(length)[0])
    for typ in range(6):
      if len(IDs[typ]) == 0:
        continue
      x,v,m, header = particles_by_ID(fileprefix_snapshot%(ss,ss),IDs[typ],opts={'pos':True,'vel':True,'mass':True})
      indices = np.arange(m.size) + 0.5
      M += np.histogram(indices,bins=subbins[:,typ],weights=m)[0]
      for d in range(3):
        Xa[:,d] += np.histogram(indices,bins=subbins[:,typ],weights=m*x[:,d])[0]
        Xb[:,d] += np.histogram(indices,bins=subbins[:,typ],weights=m*((x[:,d]+box*1./3.)%box))[0]
        Xc[:,d] += np.histogram(indices,bins=subbins[:,typ],weights=m*((x[:,d]+box*2./3.)%box))[0]
        V[:,d] += np.histogram(indices,bins=subbins[:,typ],weights=m*v[:,d])[0]
    Xa /= M[:,None]
    Xb = (Xb/M[:,None]-box*1./3.)%box
    Xc = (Xc/M[:,None]-box*2./3.)%box
    V /= M[:,None]
    for d in range(3): # should handle periodic box as long as halos are not too big
      X[:,d] = np.median([Xa[:,d],Xb[:,d],Xc[:,d]],axis=0)
    if ii == 0:
      X0, V0 = X,V

  np.savez('group_nextpos_%03d.npz'%ss0,x0=X0,v0=V0,x1=X,v1=V)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
