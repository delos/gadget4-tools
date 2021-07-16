import numpy as np
from snapshot_functions import gadget_to_particles
from numba import njit

# memory usage is 16 bytes per particle + 4 bytes per cell

# use loops for memory-efficiency
@njit
def cic_bin(x,BoxSize,GridSize,weights,density):
  NP = x.shape[1]

  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)

  hist = np.zeros((N,N,N),dtype=np.float32)
  for p in range(NP):
    f = x[:,p] / dx # (3,)

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

    hist[i[0],i[1],i[2]] += (1-f[0])*(1-f[1])*(1-f[2])*weights[p]
    hist[i1[0],i[1],i[2]] += f[0]*(1-f[1])*(1-f[2])*weights[p]
    hist[i[0],i1[1],i[2]] += (1-f[0])*f[1]*(1-f[2])*weights[p]
    hist[i[0],i[1],i1[2]] += (1-f[0])*(1-f[1])*f[2]*weights[p]
    hist[i1[0],i1[1],i[2]] += f[0]*f[1]*(1-f[2])*weights[p]
    hist[i[0],i1[1],i1[2]] += (1-f[0])*f[1]*f[2]*weights[p]
    hist[i1[0],i[1],i1[2]] += f[0]*(1-f[1])*f[2]*weights[p]
    hist[i1[0],i1[1],i1[2]] += f[0]*f[1]*f[2]*weights[p]

  if density:
    hist /= dx**3

  return hist,bins

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <grid size> [out-name]')
    return 1
  
  GridSize = int(argv[2])

  outname = 'field.bin'
  if len(argv) > 3:
    outname = argv[3]
  
  pos, mass, header = gadget_to_particles(argv[1],
    opts={'pos':True,'vel':False,'ID':False,'mass':True})
  
  BoxSize = header['BoxSize']

  mtot = mass.sum()
  print('total mass = %e'%mtot)
  print('density = %e'%(mtot/BoxSize**3))
  
  delta, bins = cic_bin(pos,BoxSize,GridSize,weights=mass,density=True)

  delta /= delta.mean()
  delta -= 1

  delta.tofile(outname)
  print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
