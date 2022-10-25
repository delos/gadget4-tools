import numpy as np
from numba import njit
from snapshot_functions import read_subhalos, particles_by_ID, fileprefix_snapshot, fileprefix_subhalo

@njit
def get_groups_subhalos(index, group_length, group_firstsub, group_numsubs, subhalo_length):
  group = np.zeros(index.size,dtype=np.int32)-1
  subhalo = np.zeros(index.size,dtype=np.int32)-1
  p = 0
  for i,gl in enumerate(group_length):
    group[(p <= index)&(index < p+gl)] = i
    q = 0
    for j,sl in enumerate(subhalo_length[group_firstsub[i]:group_firstsub[i]+group_numsubs[i]]):
      subhalo[(p+q <= index)&(index < p+q+sl)] = group_firstsub[i] + j
      q += sl
    p += gl
  return group, subhalo

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot #> <numbered peaks file> <grid N> [downscale peak-file = 1]')
    return 1

  ss = int(argv[1])
  print('  snapshot %d'%(ss))

  peakn, peaki, peakj, peakk = np.loadtxt(argv[2],usecols=(0,1,2,3),dtype=int).T
  npeak = len(peakn)
  print('  %d peaks'%npeak)

  N = int(argv[3])

  try:
    downscale = int(argv[4])
    peaki //= downscale
    peakj //= downscale
    peakk //= downscale
    print('  downscale by %d'%downscale)
  except:
    pass
  print('  range (%d,%d), (%d,%d), (%d,%d)'%(peaki.min(),peaki.max(),peakj.min(),peakj.max(),peakk.min(),peakk.max()))

  IDs = np.zeros((npeak,7),dtype=np.uint32)
  IDs[:,0] = (peaki*N + peakj)*N + peakk
  IDs[:,1] = (((peaki+1)%N)*N + peakj)*N + peakk
  IDs[:,2] = (((peaki-1)%N)*N + peakj)*N + peakk
  IDs[:,3] = (peaki*N + (peakj+1)%N)*N + peakk
  IDs[:,4] = (peaki*N + (peakj-1)%N)*N + peakk
  IDs[:,5] = (peaki*N + peakj)*N + (peakk+1)%N
  IDs[:,6] = (peaki*N + peakj)*N + (peakk-1)%N
  IDs += 1 # NGENIC convention, IDs start at 1

  print('  %d IDs (%d unique)'%(IDs.size,np.unique(IDs).size))

  groups = np.zeros(IDs.shape,dtype=np.int32)-1
  subhalos = np.zeros(IDs.shape,dtype=np.int32)-1

  ssname = fileprefix_snapshot%(ss,ss)
  grpname = fileprefix_subhalo%(ss,ss)

  _index, _type, header = particles_by_ID(ssname, IDs, opts={'index':True,'type':True})

  _sublen, _grplen, _grpfirstsub, _grpnumsubs, _ = read_subhalos(grpname,opts={'lentype':True},group_opts={'lentype':True,'firstsub':True,'numsubs':True})

  for typ in np.unique(_type):
    idx = (_type==typ)
    group, subhalo = get_groups_subhalos(_index[idx], _grplen[:,typ], _grpfirstsub, _grpnumsubs, _sublen[:,typ])
    groups[idx] = group
    subhalos[idx] = subhalo

  np.savez('gridpeak_halos_%03d.npz'%ss,snapshot=ss, peaks=peakn, IDs=IDs, groups=groups, subhalos=subhalos, )

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

