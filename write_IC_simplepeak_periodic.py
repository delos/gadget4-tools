import numpy as np
from scipy.interpolate import interp1d
from sys import argv
from IC_functions import particles_to_gadget_comoving_periodic
from snapshot_functions import gadget_to_particles

def run(argv):
  # Let's just fix these.
  H0 = 0.1 # km/s/kpc
  OmegaM = 1.0
  G = 4.3022682e-6 # (km/s)^2 kpc/Msun
  h = 0.6774
  BoxSize = 1.
  ScaleFactor = 0.03

  if len(argv) < 2:
    print('python script.py <N or glass,factor> [exp=2] [delta=0.03] [e=0.15] [p=0]')
    return 1

  try:
    N = int(argv[1])
    glassfile = glassfac = None
  except:
    N = None
    glassfile = argv[1].split(',')[0]
    glassfac = int(argv[1].split(',')[1])

  try: a = float(argv[2])
  except: a = 2.
  try: d = float(argv[3])
  except: d = 0.03
  try: e = float(argv[4])
  except: e = 0.15
  try: p = float(argv[5])
  except: p = 0.

  print('exponent=%f'%(a))
  print('delta=%.4f, e=%.4f, p=%.4f'%(d,e,p))

  l = d * np.array([1+3*e+p,1-2*p,1-3*e+p])/3. # d = np.sum(l)
  print('eigenvalues %.4f,%.4f,%.4f'%tuple(l))

  if N is None:
    print('preparing glass from %s (tiled %d)'%(glassfile,glassfac))
    pos, header = gadget_to_particles(glassfile,opts={'pos':True})
    pos = pos.T
    pos /= header['BoxSize'] * glassfac
    N3g = pos.shape[0]
    N3 = N3g*glassfac**3
    N = int(np.round(N3**(1./3)))
    pos_base = np.zeros((N3,3))
    for i in range(glassfac):
      for j in range(glassfac):
        for k in range(glassfac):
          offset = np.array([i,j,k],dtype=np.float32)/glassfac
          pos_base[N3g*((i*glassfac+j)*glassfac+k):N3g*((i*glassfac+j)*glassfac+k+1)] = pos + offset[None,:]

  print('preparing displacements, N=%d'%(N))
  grid = (np.arange(N) + 0.5) / N
  disp = grid - (4**a*grid**(1.+a))/(1.+a)
  disp[N//4:N//2] = disp[:N//4][::-1]
  disp[N//2:] = -disp[:N//2][::-1]

  if glassfile is None:
    print('preparing + displacing initial grid, N=%d'%(N))
    pos_base = np.array([x.flatten() for x in np.meshgrid(grid,grid,grid,indexing='ij')]).T
    shapes = [(-1,1,1),(1,-1,1),(1,1,-1)]
    pos = np.array([(x+disp.reshape((shapes[i]))*l[i]).flatten() for i,x in enumerate(np.meshgrid(grid,grid,grid,indexing='ij'))]).T
  else:
    print('displacing glass')
    pos = pos_base.copy()
    grid_ext = np.concatenate(([grid[-1]-1.],grid,[grid[0]+1.]))
    disp_ext = np.concatenate(([disp[-1]],disp,[disp[0]]))
    disp_interp = interp1d(grid_ext,disp_ext)
    for i in range(3):
      pos[:,i] += disp_interp(pos_base[:,i])*l[i]


  HubbleRate = H0 * np.sqrt(OmegaM * ScaleFactor**-3)
  vel = HubbleRate * (pos - pos_base)
  particles_to_gadget_comoving_periodic('init.hdf5',pos.T,vel.T,BoxSize,ScaleFactor,H0=H0,OmegaM=OmegaM,G=G)
  particles_to_gadget_comoving_periodic('pre.hdf5',pos_base.T,np.zeros_like(vel.T),BoxSize,ScaleFactor,H0=H0,OmegaM=OmegaM,G=G)

  print('Omega0                   %lg'%OmegaM)
  print('OmegaLambda              %lg'%(1-OmegaM))
  print('HubbleParam              %lg'%h) # h
  print('BoxSize                  %lg'%BoxSize)
  print('TimeBegin                %lg'%ScaleFactor)
  print('UnitLength_in_cm         3.085678e21') # 1 kpc/h
  print('UnitMass_in_g            1.989e33') # 1 Msun/h
  print('UnitVelocity_in_cm_per_s 1e5') # 1 km/s
  print('SofteningComovingClass1  %lg     ; or so'%(BoxSize/N * 0.015))

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)
