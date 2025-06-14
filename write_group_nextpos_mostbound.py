import numpy as np
from numba import njit
from snapshot_functions import subhalo_desc, read_subhalos, fileprefix_subhalo, particles_by_ID, fileprefix_snapshot

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <ss #>')
    return 1
  
  ss0 = int(argv[1])

  ss = ss0
    
  sub_IDmb, grp_sub1, header = read_subhalos(fileprefix_subhalo%(ss,ss),opts={'mostbound':True},group_opts={'firstsub':True,},)
  grp_IDmb = sub_IDmb[grp_sub1]
  
  ss = ss0 + 1
  
  x, header = particles_by_ID(fileprefix_snapshot%(ss,ss),grp_IDmb,opts={'pos':True,})

  np.save('group_nextpos_%03d.npy'%ss0,x)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
