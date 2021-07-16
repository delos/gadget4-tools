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

def lagrangian_position(ID,GridSize):
  '''
  NGENIC grid-position vs ID:

  x = ipcell / (All.GridSize * All.GridSize);
  xr = ipcell % (All.GridSize * All.GridSize);
  y = xr / All.GridSize;
  z = xr % All.GridSize;
  ID = ipcell + 1;
  '''
  ID -= 1

  x, xr = np.divmod(ID,GridSize*GridSize)
  y = np.floor_divide(xr,GridSize)
  z = np.remainder(xr,GridSize)

  return x,y,z

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <group-file,group-number> <snapshot> <gridsize> [ID most bound]')
    return 1

  ID_mostbound = None
  if len(argv) > 4:
    ID_mostbound = int(argv[4])

  grp = argv[1].split(',')
  grpfile = grp[0]
  grp = int(grp[1])

  ID = halo_IDs(argv[2],grpfile,grp)

  print('group %d -> %d particles'%(grp,ID.size))

  print(' ~ sphere of radius %.1f particles'%((3*ID.size/(4*np.pi))**(1./3)))

  GridSize = int(argv[3])

  x, y, z = lagrangian_position(ID,GridSize)

  x_cm_ = np.mean(x)
  y_cm_ = np.mean(y)
  z_cm_ = np.mean(z)

  dx_cm_ = x-x_cm_
  dy_cm_ = y-y_cm_
  dz_cm_ = z-z_cm_

  dx_cm_[dx_cm_ >= GridSize//2] -= GridSize
  dx_cm_[dx_cm_ < -GridSize//2] += GridSize
  dy_cm_[dy_cm_ >= GridSize//2] -= GridSize
  dy_cm_[dy_cm_ < -GridSize//2] += GridSize
  dz_cm_[dz_cm_ >= GridSize//2] -= GridSize
  dz_cm_[dz_cm_ < -GridSize//2] += GridSize

  x_cm = np.mean(dx_cm_) + x_cm_
  y_cm = np.mean(dy_cm_) + y_cm_
  z_cm = np.mean(dz_cm_) + z_cm_

  print('lagrangian CM = (%.1f,%.1f,%.1f) cells'%(x_cm,y_cm,z_cm))

  dx_cm = x-x_cm
  dy_cm = y-y_cm
  dz_cm = z-z_cm

  dx_cm[dx_cm >= GridSize//2] -= GridSize
  dx_cm[dx_cm < -GridSize//2] += GridSize
  dy_cm[dy_cm >= GridSize//2] -= GridSize
  dy_cm[dy_cm < -GridSize//2] += GridSize
  dz_cm[dz_cm >= GridSize//2] -= GridSize
  dz_cm[dz_cm < -GridSize//2] += GridSize

  r_cm = np.sqrt(dx_cm**2+dy_cm**2+dz_cm**2)

  print('max radius = %.1f cells'%(r_cm.max()))
  print('rms radius = %.1f cells'%(np.sqrt(np.mean(r_cm**2))))

  if ID_mostbound is not None:
    x_mb, y_mb, z_mb = lagrangian_position(ID_mostbound,GridSize)

    print('lagrangian MB = (%d,%d,%d) cells'%(x_mb,y_mb,z_mb))

    dx_mb = x-float(x_mb)
    dy_mb = y-float(y_mb)
    dz_mb = z-float(z_mb)

    dx_mb[dx_mb >= GridSize//2] -= GridSize
    dx_mb[dx_mb < -GridSize//2] += GridSize
    dy_mb[dy_mb >= GridSize//2] -= GridSize
    dy_mb[dy_mb < -GridSize//2] += GridSize
    dz_mb[dz_mb >= GridSize//2] -= GridSize
    dz_mb[dz_mb < -GridSize//2] += GridSize

    r_mb = np.sqrt(dx_mb**2+dy_mb**2+dz_mb**2)

    print('max radius = %.1f cells'%(r_mb.max()))
    print('rms radius = %.1f cells'%(np.sqrt(np.mean(r_mb**2))))

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

