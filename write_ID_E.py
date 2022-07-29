import numpy as np
from numba import njit
from snapshot_functions import group_data, read_particles_filter

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
  
  if len(argv) < 3:
    print('python script.py <group-file,group-number> <snapshot> [out-file]')
    return 1

  outfile = 'ID_E.npz'
  if len(argv) > 3:
    outfile = argv[3]

  snapfile = argv[2]

  grp = argv[1].split(',')
  grpfile = grp[0]
  grp = int(grp[1])

  gpos, gvel, grad, header = group_data(grpfile,grp,size_definition='Mean200',opts={'pos':True,'vel':True,'radius':True,})

  pos, vel, ID, pot, header = read_particles_filter(snapfile,center=gpos,radius=grad,
    opts={'pos':True,'vel':True,'ID':True,'pot':True})
  print('group %d : %d particles'%(grp,ID.size))

  a = 1./(1+header['Redshift'])

  vel -= gvel[None,:]
  vel += a*pos # peculiar -> physical

  pot /= a # comoving -> physical
  T = 0.5*np.sum(vel**2,axis=1)

  E = T + pot

  #np.savez(outfile,ID=ID,E=E)
  np.savez(outfile,ID=ID,E=E,V=pot,T=T)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

