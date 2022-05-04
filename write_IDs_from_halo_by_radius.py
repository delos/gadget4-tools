import numpy as np
from numba import njit
from snapshot_functions import group_extent, read_particles_filter

@njit
def reduce_IDs(rmin,rmax,pos,ID):
  ID_index = 0
  for i in range(pos.shape[1]):
    if rmin**2 <= np.sum(pos[:,i]**2) < rmax**2:
      ID[ID_index] = ID[i]
      ID_index += 1
  return ID_index

def halo_IDs(snap,groupfile,group,rmin,rmax):
  gpos, grad, header = group_extent(groupfile,group)
  print('group position = (%.6e,%.6e,%.6e)'%tuple(gpos))
  print('group radius = %.6e'%grad)

  # get positions centered on group
  pos, ID, _ = read_particles_filter(snap,center=gpos,radius=rmax,
    opts={'pos':True,'vel':False,'ID':True,'mass':False})

  # cut IDs whose positions lie outside group radius
  count = reduce_IDs(rmin,rmax,pos,ID)
  ID.resize(count,refcheck=False)

  return ID

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <group-file,group-number> <snapshot> <r min,max> [out-file]')
    return 1

  outfile = 'ID.bin'
  if len(argv) > 4:
    outfile = argv[4]

  grp = argv[1].split(',')
  grpfile = grp[0]
  grp = int(grp[1])

  rmin,rmax = [float(x) for x in argv[3].split(',')]

  ID = halo_IDs(argv[2],grpfile,grp,rmin,rmax)

  print('group %d -> %d particles'%(grp,ID.size))

  ID.tofile(outfile)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

