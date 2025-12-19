import numpy as np
from numba import njit
from snapshot_functions import read_subhalos, read_particles_filter

def center(ID,GridSize):

  # ID to grid position
  x, xr = np.divmod(ID-1,GridSize*GridSize)
  y, z  = np.divmod(xr,GridSize)

  x = x.astype(np.int32)
  y = y.astype(np.int32)
  z = z.astype(np.int32)

  # pick one and center about it
  x0 = x[0]
  y0 = y[0]
  z0 = z[0]

  x -= x0
  y -= y0
  z -= z0

  x[x<-GridSize//2] += GridSize
  y[y<-GridSize//2] += GridSize
  z[z<-GridSize//2] += GridSize
  x[x>=GridSize//2] -= GridSize
  y[y>=GridSize//2] -= GridSize
  z[z>=GridSize//2] -= GridSize

  # get center of mass
  xc = int(np.round(np.mean(x) + x0))
  yc = int(np.round(np.mean(y) + y0))
  zc = int(np.round(np.mean(z) + z0))

  # now center positions about the center of mass
  x -= xc-x0
  y -= yc-y0
  z -= zc-z0

  return np.array([x,y,z]).T, np.array([xc,yc,zc]) # (N,3), (3,)

def sphere(ID, GridSize):
  x, xc = center(ID, GridSize)
  r = np.sum(x**2,axis=1)**0.5
  R = r.max()
  print('Lagrangian radius = %.1f cells'%R)
  Rint = int(np.ceil(R))
  grid = np.arange(-Rint,Rint+1)
  X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')
  X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
  mask = X*X + Y*Y + Z*Z <= R*R
  X, Y, Z = X[mask]+xc[0], Y[mask]+xc[1], Z[mask]+xc[2]
  return ((X*GridSize + Y)*GridSize + Z + 1).astype(np.uint32)

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <group-file,group-number> <snapshot> <grid size> [out-file]')
    return 1

  outfile = 'ID.bin'
  if len(argv) > 4:
    outfile = argv[4]

  grp = argv[1].split(',')
  grpfile = grp[0]
  grp = int(grp[1])

  grpl, header = read_subhalos(grpfile,opts={},group_opts={'lentype':True})

  if grp == 0:
    i0 = np.zeros(6,dtype=np.uint32)
  else:
    i0 = np.sum(grpl[:grp],axis=0)
  i1 = i0 + grpl[grp]

  ID, header = read_particles_filter(argv[2],part_range=(i0,i1),opts={'ID':True})

  print('group %d has %d particles'%(grp,ID.size))

  GridSize = int(argv[3])
  ID = sphere(ID,GridSize)

  print('%d particles in Lagrangian sphere'%(ID.size))

  ID.tofile(outfile)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

