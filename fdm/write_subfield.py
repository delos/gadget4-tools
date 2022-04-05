import numpy as np
from numba import njit
from pathlib import Path
import h5py

@njit
def subfield_part(N,target_field,input_array,input_inst,input_count,ilim,jlim,klim):
  N3_per_inst = len(input_array)
  for idx in range(N3_per_inst*input_inst,N3_per_inst*(input_inst+1)):
    i = idx // (N*N)
    j = (idx % (N*N)) // N
    k = idx % N
    if i >= ilim[0] and i <= ilim[1] and j >= jlim[0] and j <= jlim[1] and k >= klim[0] and k <= klim[1]:
      target_field[i-ilim[0],j-jlim[0],k-klim[0]] = input_array[idx - N3_per_inst*input_inst]

@njit
def subfield_indices(N,N3_per_inst,input_inst,input_count,ilim,jlim,klim):
  size = ilim[1]-ilim[0]+1
  out = np.zeros((size,size,size),dtype=np.int32) - 1

  for idx in range(N3_per_inst*input_inst,N3_per_inst*(input_inst+1)):
    i = idx // (N*N)
    j = (idx % (N*N)) // N
    k = idx % N
    if i >= ilim[0] and i <= ilim[1] and j >= jlim[0] and j <= jlim[1] and k >= klim[0] and k <= klim[1]:
      out[i-ilim[0],j-jlim[0],k-klim[0]] = idx - N3_per_inst*input_inst
  return out

@njit
def skip_inst(N,N3_per_inst,input_inst,input_count,ilim):
  if N3_per_inst % (N**2) == 0:
    Ni_per_inst = N3_per_inst // (N**2)
    imin = Ni_per_inst*input_inst
    imax = Ni_per_inst*(input_inst+1) - 1
    if not (imax >= ilim[0] and imin <= ilim[1]):
      return True
  return False

def read_subfield(fileprefix, center, halfwidth):

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
  while fileinst < numfiles:

    if fileinst > 0 and skip_inst(N,N3_per_inst,fileinst,numfiles,ilim):
      fileinst += 1
      continue

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      MassTable = header['MassTable']
      ScaleFactor = 1./(1+header['Redshift'])
      GridSize = header['NumPart_Total'][1]
      BoxSize = header['BoxSize']
      numfiles = header['NumFilesPerSnapshot']

      if fileinst == 0:
        # allocate full-sized memory block in advance, for efficiency
        size = 2*int(halfwidth / BoxSize * GridSize + .5)
        field = np.zeros((size,size,size),dtype=np.float32)
        # (center-halfwidth,center+halfwidth) = BoxSize/GridSize * (lim[0]-0.5, lim[1]+0.5)
        ilim = (int((center[0]-halfwidth)/BoxSize * GridSize + 1),int((center[0]+halfwidth)/BoxSize * GridSize ))
        jlim = (int((center[1]-halfwidth)/BoxSize * GridSize + 1),int((center[1]+halfwidth)/BoxSize * GridSize ))
        klim = (int((center[2]-halfwidth)/BoxSize * GridSize + 1),int((center[2]+halfwidth)/BoxSize * GridSize ))
        N3_per_inst = f['FuzzyDM/PsiRe'].size
        N3 = f['FuzzyDM/PsiRe'].size*numfiles
        N = int(N3**(1./3) + .5)

        print(size,ilim,jlim,klim,N3,N,GridSize)

        if skip_inst(N,N3_per_inst,fileinst,numfiles,ilim):
          fileinst += 1
          continue

      density = np.array(f['FuzzyDM/PsiRe'])**2 + np.array(f['FuzzyDM/PsiIm'])**2
      subfield_part(N,field,density,fileinst,numfiles,ilim,jlim,klim)
      '''
      indices = subfield_indices(N,N3_per_inst,fileinst,numfiles,ilim,jlim,klim)
      valid = indices >= 0
      count = np.sum(valid)
      if count > 0:
        print('  %d cells'%count)
        print(density)
        field[valid] = density[valid]
      '''
    fileinst += 1

  return field, header

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot> <x,y,z,r> [out-name]')
    return 1

  filename = argv[1]
  
  x,y,z,r = [float(x) for x in argv[2].split(',')]
  center = [x,y,z]
  w = 2. * r

  try: outname = argv[3]
  except: outname = filename + '.field'
  
  # read
  dens, header = read_subfield(filename, center, r)

  dens.tofile(outname)
  print('saved to ' + outname)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
