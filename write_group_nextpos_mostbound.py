import numpy as np
from numba import njit
from snapshot_functions import subhalo_desc, read_subhalos, fileprefix_subhalo, particles_by_ID, fileprefix_snapshot

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <ss start>')
    return 1
  
  ss0 = int(argv[1])

  ss = ss0
  while True:
    if ss > ss0:
      try:
        x, header = particles_by_ID(fileprefix_snapshot%(ss,ss),grp_IDmb,opts={'pos':True,})
      except Exception as e:
        print(e)
        break
      np.save('group_nextpos_%03d.npy'%(ss-1),x)
    
    try:
      sub_IDmb, grp_sub1, header = read_subhalos(fileprefix_subhalo%(ss,ss),opts={'mostbound':True},group_opts={'firstsub':True,},)
    except Exception as e:
      print(e)
      break
    grp_IDmb = sub_IDmb[grp_sub1]
    ss += 1

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
