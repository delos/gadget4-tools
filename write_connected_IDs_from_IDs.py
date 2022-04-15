import numpy as np
from numba import njit

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

@njit
def connect(x):
  x_ = [x]

  while x_[-1][0] != 0 or x_[-1][1] != 0 or x_[-1][2] != 0:
    d = np.argmax(np.abs(x_[-1]))
    x_ += [x_[-1].copy()]
    x_[-1][d] -= np.sign(x_[-1][d])

  return x_

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <ID-file> <n> [out-file]')
    return 1
  
  GridSize = int(argv[2])

  outfile = argv[1] + '_ext'
  if len(argv) > 3:
    outfile = argv[3]

  ID = np.fromfile(argv[1],dtype=np.uint32)

  x, xc = center(ID,GridSize) # box positions are (x+xc)%GridSize

  print('center at (%d,%d,%d)'%tuple(xc))
  print('original extent: (%d,%d,%d) to (%d,%d,%d)'%(
    (x+xc)[:,0].min(),(x+xc)[:,1].min(),(x+xc)[:,2].min(),
    (x+xc)[:,0].max(),(x+xc)[:,1].max(),(x+xc)[:,2].max(),
    ))

  print('start with %d cells'%x.shape[0])

  x_ext = []
  for i in range(x.shape[0]):
    x_ext += connect(x[i])

  print('%d path cells'%len(x_ext))

  x_ext = np.unique(x_ext,axis=0)

  print('of which %d are unique'%(x_ext.shape[0]))

  x_ext += xc.reshape((1,3))

  print('final extent: (%d,%d,%d) to (%d,%d,%d)'%(
    x_ext[:,0].min(),x_ext[:,1].min(),x_ext[:,2].min(),
    x_ext[:,0].max(),x_ext[:,1].max(),x_ext[:,2].max(),
    ))
  print('(%.3f,%.3f,%.3f) of box'%(
    (x_ext[:,0].max()-x_ext[:,0].min())/GridSize,
    (x_ext[:,1].max()-x_ext[:,1].min())/GridSize,
    (x_ext[:,2].max()-x_ext[:,2].min())/GridSize,
    ))

  x_ext[x_ext < 0] += GridSize
  x_ext[x_ext >= GridSize] -= GridSize

  x_ext = x_ext.astype(np.uint32)

  ID_ext = (x_ext[:,0]*GridSize + x_ext[:,1])*GridSize + x_ext[:,2] + 1

  ID_ext.tofile(outfile)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

