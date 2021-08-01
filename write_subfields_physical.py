import numpy as np
from numba import njit
from snapshot_functions import list_snapshots
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

def read_positions(fileprefix, xlim,ylim,zlim):
  filepath = [
    Path(fileprefix + '.hdf5'),
    Path(fileprefix + '.0.hdf5'),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d.hdf5'
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1

  fileinst = 0
  pinst = 0
  pos = []
  mass = []
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      MassTable = header['MassTable']
      ScaleFactor = 1./(1+header['Redshift'])
      NP = header['NumPart_ThisFile']
      NPtot = header['NumPart_Total']
      numfiles = header['NumFilesPerSnapshot']

      for typ in range(len(NPtot)):
        NPtyp = int(NP[typ])
        if NPtyp == 0:
          continue

        pos_ = np.array(f['PartType%d/Coordinates'%typ])

        idx =     (xlim[0]<=pos_[:,0])&(xlim[1]>=pos_[:,0])
        idx = idx&(ylim[0]<=pos_[:,1])&(ylim[1]>=pos_[:,1])
        idx = idx&(zlim[0]<=pos_[:,2])&(zlim[1]>=pos_[:,2])

        pos += [pos_[idx]]

        if MassTable[typ] == 0.:
          mass_ = np.array(f['PartType%d/Masses'%typ])
        else:
          mass_ = np.full(NPtyp,MassTable[typ])

        mass += [mass_[idx]]

        pinst += NPtyp

    fileinst += 1

  return np.concatenate(pos,axis=0), np.concatenate(mass), header

def run(argv):
  
  if len(argv) < 5:
    print('python script.py <snapshot min,max> <grid size> <x,y,z> <r> [out-name]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]
  
  GridSize = int(argv[2])

  x,y,z = [float(x) for x in argv[3].split(',')]

  r_phys = float(argv[4])

  outbase = 'subfield'
  if len(argv) > 5:
    outbase = argv[5]
  
  names, headers = list_snapshots()

  for i, filename in enumerate(names):
    snapshot_number = int(filename[-3:])

    if snapshot_number < ssmin or snapshot_number > ssmax:
      continue

    outname = outbase + filename[-4:] + '.bin'
    scale = headers[i]['Time']
    r = r_phys / scale

    pos, mass, header = read_positions(filename,[x-r,x+r],[y-r,y+r],[z-r,z+r])

    pos -= np.array([x-r,y-r,z-r]).reshape((1,3))

    dens, bins = cic_bin(pos,2*r,GridSize,weights=mass,density=True)

    dens /= scale**3

    dens.tofile(outname)
    print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
