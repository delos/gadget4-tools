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
  
  if len(argv) < 3:
    print('python script.py <snapshot #> <ID>')
    return 1
  
  ss = int(argv[1])
  ID = int(argv[2])
  
  ssname = fileprefix_snapshot%(ss,ss)
  grpname = fileprefix_subhalo%(ss,ss)

  pos, _index, _type, header = particles_by_ID(ssname, np.array([ID]), opts={'pos':True,'index':True,'type':True})

  _sublen, _grplen, _grpfirstsub, _grpnumsubs, _ = read_subhalos(grpname,opts={'lentype':True},group_opts={'lentype':True,'firstsub':True,'numsubs':True})

  group, subhalo = get_groups_subhalos(_index, _grplen[:,_type[0]], _grpfirstsub, _grpnumsubs, _sublen[:,_type[0]])

  print('snapshot %d, particle %d'%(ss,ID))
  print('(%g,%g,%g), group %d, subhalo %d'%(pos[0,0],pos[0,1],pos[0,2],group[0],subhalo[0]))

  return 0

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

