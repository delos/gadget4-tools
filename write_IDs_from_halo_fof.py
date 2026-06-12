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

def minimum_enclosing_ellipsoid(x, y, z, tol=1e-7, max_iter=1000):
    """
    Compute the minimum volume enclosing ellipsoid of 3D points.

    Parameters
    ----------
    x, y, z : array-like, shape (N,)
        Coordinates of points
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    center : ndarray, shape (3,)
        Center of ellipsoid
    A : ndarray, shape (3, 3)
        Ellipsoid matrix such that (p-center)^T A (p-center) <= 1
    """
    P = np.vstack([x, y, z]).T  # (N, 3)
    N, d = P.shape

    # Homogeneous coordinates
    Q = np.vstack([P.T, np.ones(N)])  # (4, N)

    # Initial weights
    u = np.ones(N) / N

    for _ in range(max_iter):
        # Compute matrix X
        X = Q @ np.diag(u) @ Q.T
        X_inv = np.linalg.inv(X)

        # Compute Mahalanobis distances
        M = np.einsum('ij,jk,ki->i', Q.T, X_inv, Q)

        j = np.argmax(M)
        max_M = M[j]

        step = (max_M - d - 1) / ((d + 1) * (max_M - 1))
        new_u = (1 - step) * u
        new_u[j] += step

        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break

        u = new_u

    # Ellipsoid center
    center = P.T @ u

    # Ellipsoid matrix
    Pu = P - center
    cov = Pu.T @ np.diag(u) @ Pu
    A = np.linalg.inv(cov) / d

    return center, A # Ellipsoid matrix such that (p-center)^T A (p-center) <= 1

def sphere(ID, GridSize,tol=0.01):

  # center particles about origin and find enclosing sphere
  x, xc = center(ID, GridSize)
  r = np.sum(x**2,axis=1)**0.5
  R = r.max()
  print('Lagrangian sphere: %.1f cells radius about (%d,%d,%d)'%(R,xc[0],xc[1],xc[2]))

  # find enclosing ellipsiod
  c, A = minimum_enclosing_ellipsoid(x[:,0],x[:,1],x[:,2])

  # select cells
  Rint = int(np.ceil(R))
  grid = np.arange(-Rint,Rint+1)
  X = np.array(np.meshgrid(grid,grid,grid,indexing='ij')).reshape((3,-1)).T # (N,3)
  mask = (X[:,None,:]-c[None,None])@A[None]@(X[:,:,None]-c[None,:,None]) <= (1.+tol)**2
  mask = mask[:,0,0]
  Xe = (X[mask] + xc[None])%GridSize
  return ((Xe[:,0]*GridSize + Xe[:,1])*GridSize + Xe[:,2] + 1).astype(np.uint32)

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

  tol = 0.000
  while True:
    tol += 0.001
    ID_enclosing = sphere(ID,GridSize,tol)
    print('%d particles in Lagrangian ellipsoid'%(ID_enclosing.size))
    if np.all(np.isin(ID,ID_enclosing)):
      break
    print('  not entirely in Lagrangian ellipsoid, try again')

  ID_enclosing.tofile(outfile)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

