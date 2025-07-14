import numpy as np
from snapshot_functions import read_particles_filter
from numba import njit

# memory usage is 16 bytes per particle + 4 bytes per cell

# use loops for memory-efficiency
@njit
def cic_bin(x,BoxSize,GridSize,weights,density):
  NP = x.shape[0]

  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)

  hist = np.zeros((N,N),dtype=np.float32)
  for p in range(NP):
    f = x[p,:2] / dx # (3,)

    i = np.floor(f-0.5).astype(np.int32)
    i1 = i+1
    
    for d in range(2):
      f[d] -= i[d] + 0.5

    if 0 <= i[0] and 0 <= i[1]:
      hist[i[0],i[1]] += (1-f[0])*(1-f[1])*weights[p]
    if i1[0] < N and 0 <= i[1]:
      hist[i1[0],i[1]] += f[0]*(1-f[1])*weights[p]
    if 0 <= i[0] and i1[1] < N:
      hist[i[0],i1[1]] += (1-f[0])*f[1]*weights[p]
    if i1[0] < N and i1[1] < N:
      hist[i1[0],i1[1]] += f[0]*f[1]*weights[p]

  if density:
    hist /= dx**2

  return hist,bins

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <gridsize,axis> <x,y,z,r> [types=-1] [out-name]')
    return 1
  print(argv)

  filename = argv[1]
  
  GridSize, axis = [int(x) for x in argv[2].split(',')]

  x,y,z,r = [float(x) for x in argv[3].split(',')]
  center = [x,y,z]
  w = 2. * r

  try:
    types = [int(x) for x in argv[4].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  rotation = None
  part_range = None

  try: outname = argv[5]
  except: outname = filename + '.field'
  
  pos, mass, header = read_particles_filter(filename,center=center,halfwidth=r,rotation=rotation,type_list=types,part_range=part_range,opts={'mass':True,'pos':True})
  print('%d particles'%len(mass) + ' in %g (Mpc/h)^3'%(w**3))
  
  pos += np.array([r,r,r]).reshape((1,3))

  dens, bins = cic_bin(np.roll(pos,2-axis,axis=1),w,GridSize,weights=mass,density=True)

  dens.tofile(outname)
  print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
