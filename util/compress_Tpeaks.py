import numpy as np
import h5py
from sys import argv

grid = 4096

filebase = 'Tpeaks_%.7f_%04d.txt'
r_list = np.geomspace(1./grid,0.25,256)

filepeaks = 'field%d_numberedpeaks.txt'%grid
fileparams = 'peak_parameters.txt'

with h5py.File('Tpeaks.hdf5', 'w') as f:
  data = np.loadtxt(filepeaks,dtype=int,usecols=(0,1,2,3,))

  # initial locations
  group = f.create_group('Initial')
  group.create_dataset('n', data=data[:,0])
  group.create_dataset('x', data=data[:,1:4]/grid)

  # initial properties
  data = np.loadtxt(fileparams,dtype=int,usecols=(0,))
  sort = np.argsort(data)
  data = np.loadtxt(fileparams,dtype=np.float32,usecols=(1,2,3,4,))
  group.create_dataset('d',   data=data[:,0][sort])
  group.create_dataset('-d2d',data=data[:,1][sort])
  group.create_dataset('e',   data=data[:,2][sort])
  group.create_dataset('p',   data=data[:,3][sort])

  for i,r in enumerate(r_list):
    print(i,r)
    group = f.create_group('Radius%d'%i)
    group.attrs['r'] = np.array(r,dtype=np.float32)
    x = []
    d = []
    e = []
    p = []
    j = 0
    while True:
      try:
        data = np.loadtxt(filebase%(r,j))
      except:
        break
      if data.size == 0:
        j += 1
        continue
      if np.size(data.shape) == 1:
        data.shape = (1,-1)
      print('  reading ' + filebase%(r,j))
      x += [data[:,0:3]/grid]
      d += [data[:,3]]
      e += [data[:,4]]
      p += [data[:,5]]
      j += 1
    x = np.concatenate(x,axis=0,dtype=np.float32)
    d = np.concatenate(d,dtype=np.float32)
    e = np.concatenate(e,dtype=np.float32)
    p = np.concatenate(p,dtype=np.float32)
    group.create_dataset('x', data=x)
    group.create_dataset('d', data=d)
    group.create_dataset('e', data=e)
    group.create_dataset('p', data=p)

