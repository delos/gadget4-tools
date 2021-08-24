import numpy as np
from numba import njit
from snapshot_functions import list_snapshots, read_particles_filter
from pathlib import Path
import h5py

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
  
  if len(argv) < 5:
    print('python script.py <snapshot min,max> <grid size> <x,y,z or trace-file> <r> [rotation-file=0] [physical=0] [out-name]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]
  
  GridSize = int(argv[2])

  try:
    _data = np.loadtxt(argv[3])
    _ss = _data[:,0]
    _x = _data[:,2]
    _y = _data[:,3]
    _z = _data[:,4]
    print('using trace file')
  except Exception as e:
    print('using coordinates')
    x,y,z = [float(x) for x in argv[3].split(',')]
  
  r = float(argv[4])

  rotation = None
  if len(argv) > 5:
    try:
      rotation = np.loadtxt(argv[5])
      print('rotation matrix:')
      with np.printoptions(precision=3, suppress=True):
        print(rotation)
    except:
      rotation = None
      print('no rotation')

  phys = False
  if len(argv) > 6:
    if argv[6][0].lower() == 't': phys = True
    elif argv[6][0].lower() == 'f': phys = False
    elif int(argv[6]) != 0: phys = True
  if phys:
    r_phys = r
    print('transform to physical')

  outbase = 'subfield'
  if len(argv) > 7:
    outbase = argv[7]
  
  names, headers = list_snapshots()

  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])

    if snapshot_number < ssmin or snapshot_number > ssmax:
      continue

    outname = outbase + filename[-4:] + '.bin'
    scale = headers[i]['Time']

    try:
      if snapshot_number < _ss[0]:
        center = [_x[0],_y[0],_z[0]]
      elif snapshot_number > _ss[-1]:
        center = [_x[-1],_y[-1],_z[-1]]
      else:
        center = [
          _x[_ss==snapshot_number][0],
          _y[_ss==snapshot_number][0],
          _z[_ss==snapshot_number][0],
          ]
    except:
      center = [x,y,z]

    if phys:
      r = r_phys / scale

    pos, mass, header = read_particles_filter(filename,center=center,halfwidth=r,opts={'mass':True,'pos':True})

    pos += np.array([r,r,r]).reshape((1,3))

    dens, bins = cic_bin(pos,2*r,GridSize,weights=mass,density=True)

    if phys:
      dens /= scale**3

    dens.tofile(outname)
    print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
