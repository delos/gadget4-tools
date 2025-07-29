import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter, read_subhalos, fileprefix_snapshot, fileprefix_subhalo

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
    print('python script.py <snapshot min,max[,directory]> <ID>')
    return 1

  ID = int(argv[2])

  ssargs = argv[1].split(',')
  ssmin,ssmax = [int(x) for x in ssargs[:2]]
  if len(ssargs) > 2:
    direc = ssargs[2] + '/'
  else:
    direc = ''

  with open('trace_%d.txt'%(ID),'wt') as f:
    f.write('# snapshot, a, x, y, z\n')
    for ss in range(ssmin,ssmax+1):

      try:
        _index, _type, header = read_particles_filter(direc + fileprefix_snapshot%(ss,ss),ID_list=[ID],
          opts={'index':True,'type':True})
        _subx, _subv, _subM, _sublen, _grplen, _grpfirstsub, _grpnumsubs, _ = read_subhalos(direc + fileprefix_subhalo%(ss,ss),
          opts={'pos':True,'vel':True,'mass':True,'lentype':True},group_opts={'lentype':True,'firstsub':True,'numsubs':True})
      except:
        _index, _type, header = read_particles_filter(direc + (fileprefix_snapshot%(ss,ss)).split('/')[-1],ID_list=[ID],
          opts={'index':True,'type':True})
        _subx, _subv, _subM, _sublen, _grplen, _grpfirstsub, _grpnumsubs, _ = read_subhalos(direc + (fileprefix_subhalo%(ss,ss)).split('/')[-1],
          opts={'pos':True,'vel':True,'mass':True,'lentype':True},group_opts={'lentype':True,'firstsub':True,'numsubs':True})

      _typ = _type[0]
      group, subhalo = get_groups_subhalos(_index, _grplen[:,_typ], _grpfirstsub, _grpnumsubs, _sublen[:,typ])
      subM = _subM[subhalo]
      subx = _subx[subhalo]
      subv = _subv[subhalo]
      
      f.write('%d %.6e %.6e %.6e %.6e  %.6e %.6e %.6e %.6e %d\n'%(ss, header['Time'], subx[0,0], subx[0,1], subx[0,2], subv[0,0], subv[0,1], subv[0,2], subM[0], subhalo[0]))

  print(pos.tolist())

  return 0

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

