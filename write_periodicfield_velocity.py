import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter

# memory usage is 16 bytes per particle + 4 bytes per cell

# use loops for memory-efficiency
@njit
def cic_bin(x,BoxSize,GridSize,weights,density):
  NP = x.shape[0]

  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)

  hist = np.zeros((N,N,N),dtype=np.float32)
  for p in range(NP):
    f = x[p] / dx # (3,)

    for d in range(3):
      if f[d] < 0.5: f[d] += N
      if f[d] >= N+0.5: f[d] -= N

    i = (f-0.5).astype(np.int32)
    i1 = i+1

    for d in range(3):
      f[d] -= i[d] + 0.5

      if i[d] < 0: i[d] += N
      if i[d] >= N: i[d] -= N
      if i1[d] < 0: i1[d] += N
      if i1[d] >= N: i1[d] -= N

    if  i[0] >= 0 and  i[0] < N and  i[1] >= 0 and  i[1] < N and  i[2] >= 0 and  i[2] < N:
      hist[i[0],i[1],i[2]] += (1-f[0])*(1-f[1])*(1-f[2])*weights[p]
    if i1[0] >= 0 and i1[0] < N and  i[1] >= 0 and  i[1] < N and  i[2] >= 0 and  i[2] < N:
      hist[i1[0],i[1],i[2]] += f[0]*(1-f[1])*(1-f[2])*weights[p]
    if  i[0] >= 0 and  i[0] < N and i1[1] >= 0 and i1[1] < N and  i[2] >= 0 and  i[2] < N:
      hist[i[0],i1[1],i[2]] += (1-f[0])*f[1]*(1-f[2])*weights[p]
    if  i[0] >= 0 and  i[0] < N and  i[1] >= 0 and  i[1] < N and i1[2] >= 0 and i1[2] < N:
      hist[i[0],i[1],i1[2]] += (1-f[0])*(1-f[1])*f[2]*weights[p]
    if i1[0] >= 0 and i1[0] < N and i1[1] >= 0 and i1[1] < N and  i[2] >= 0 and  i[2] < N:
      hist[i1[0],i1[1],i[2]] += f[0]*f[1]*(1-f[2])*weights[p]
    if  i[0] >= 0 and  i[0] < N and i1[1] >= 0 and i1[1] < N and i1[2] >= 0 and i1[2] < N:
      hist[i[0],i1[1],i1[2]] += (1-f[0])*f[1]*f[2]*weights[p]
    if i1[0] >= 0 and i1[0] < N and  i[1] >= 0 and  i[1] < N and i1[2] >= 0 and i1[2] < N:
      hist[i1[0],i[1],i1[2]] += f[0]*(1-f[1])*f[2]*weights[p]
    if i1[0] >= 0 and i1[0] < N and i1[1] >= 0 and i1[1] < N and i1[2] >= 0 and i1[2] < N:
      hist[i1[0],i1[1],i1[2]] += f[0]*f[1]*f[2]*weights[p]

  if density:
    hist /= dx**3

  return hist,bins

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot> <grid size> [types=-1] [out-name]')
    return 1

  filename = argv[1]
  
  GridSize = int(argv[2])

  try:
    types = [int(x) for x in argv[3].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  try: outname = argv[4]
  except: outname = filename + '.field'
  
  try: contrast = int(argv[5])
  except: contrast = False

  # read
  pos, vel, mass, header = read_particles_filter(filename,type_list=types,
    opts={'mass':True,'pos':True,'vel':True})
  print('%d particles'%len(mass) + ' in volume %g'%(header['BoxSize']**3))

  dens, bins = cic_bin(pos,header['BoxSize'],GridSize,weights=mass,density=False)

  vx, _ = cic_bin(pos,header['BoxSize'],GridSize,weights=mass*vel[:,0],density=False)
  vy, _ = cic_bin(pos,header['BoxSize'],GridSize,weights=mass*vel[:,1],density=False)
  vz, _ = cic_bin(pos,header['BoxSize'],GridSize,weights=mass*vel[:,2],density=False)

  vx /= dens
  vy /= dens
  vz /= dens

  vx[dens==0.] = 0.
  vy[dens==0.] = 0.
  vz[dens==0.] = 0.

  vx.tofile(outname+'x')
  vy.tofile(outname+'y')
  vz.tofile(outname+'z')
  print('saved to %s, %s, %s'%(outname+'x',outname+'y',outname+'z'))

if __name__ == '__main__':
  from sys import argv
  run(argv)
