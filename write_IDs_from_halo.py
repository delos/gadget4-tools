import numpy as np
from numba import njit
from snapshot_functions import group_extent, gadget_to_particles

# loops are more memory-efficient than array indexing operations
@njit
def center_periodic(pos,BoxSize):
  for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
      if pos[i,j] >= 0.5*BoxSize:
        pos[i,j] -= BoxSize
      if pos[i,j] < -0.5*BoxSize:
        pos[i,j] += BoxSize
@njit
def reduce_IDs(rad,pos,ID):
  ID_index = 0
  for i in range(pos.shape[1]):
    if np.sum(pos[:,i]**2) <= rad**2:
      ID[ID_index] = ID[i]
      ID_index += 1
  return ID_index

def halo_IDs(snap,groupfile,group):
  gpos, grad, header = group_extent(groupfile,group)
  print('group position = (%.6e,%.6e,%.6e)'%tuple(gpos))
  print('group radius = %.6e'%grad)

  pos, ID, _ = gadget_to_particles(snap,opts={'pos':True,'vel':False,'ID':True,'mass':False})
  pos -= gpos[:,True]

  center_periodic(pos,header['BoxSize'])

  count = reduce_IDs(grad,pos,ID)
  ID.resize(count,refcheck=False)

  return ID

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <group-file,group-number> <snapshot> [out-file]')
    return 1

  outfile = 'ID.bin'
  if len(argv) > 3:
    outfile = argv[3]

  grp = argv[1].split(',')
  grpfile = grp[0]
  grp = int(grp[1])

  ID = halo_IDs(argv[2],grpfile,grp)

  print('group %d -> %d particles'%(grp,ID.size))

  ID.tofile(outfile)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

