import numpy as np
from numba import njit
from snapshot_functions import read_subhalos, fileprefix_subhalo, particles_by_ID, fileprefix_snapshot, read_particles_filter

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <ss start>')
    return 1
  
  ss0 = int(argv[1])

  ss = ss0
  while True:
    if ss > ss0:
      try:
        X = np.zeros((np.shape(bins)[0]-1,3))
        V = np.zeros((np.shape(bins)[0]-1,3))
        M = np.zeros(np.shape(bins)[0]-1)
        for typ in range(6):
          x,v,m, header = particles_by_ID(fileprefix_snapshot%(ss,ss),IDs[typ],opts={'pos':True,'vel':True,'mass':True})
          indices = np.arange(m.size)
          for d in range(3):
            X[:,d] += np.histogram(indices,bins=bins[:,typ],weights=m*x[:,d])[0]
            V[:,d] += np.histogram(indices,bins=bins[:,typ],weights=m*v[:,d])[0]
          M += np.histogram(indices,bins=bins[:,typ],weights=m)[0]
        X /= M[:,None]
        V /= M[:,None]
      except Exception as e:
        print(e)
        break
      np.savez('group_nextpos_%03d.npz'%(ss-1),x=X,v=V)
    
    try:
      length, header = read_subhalos(fileprefix_subhalo%(ss,ss),opts={},group_opts={'lentype':True,},)
      bins = np.concatenate((np.zeros(6,dtype=int)[None],np.cumsum(length,axis=0)))
      IDs = []
      for typ in range(6):
        ID, header = read_particles_filter(fileprefix_snapshot%(ss,ss),type_list=[typ],opts={'ID':True},part_range=(bins[0],bins[-1]))
        IDs += [ID]
    except Exception as e:
      print(e)
      break
    ss += 1

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
