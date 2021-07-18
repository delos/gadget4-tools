import numpy as np
from numba import njit
import h5py
from snapshot_functions import gadget_to_particles

def read_glass(filename):

  rng = np.random.default_rng(38974)

  pos, header = gadget_to_particles(filename,
    opts={'pos':True,'vel':False,'ID':False,'mass':False})
  pos /= header['BoxSize']

  # ensure CM of glass is centered
  pos -= np.mean(pos,axis=1)[:,None] - 0.5
  while np.any((pos>=1.)|(pos<0.)):
    pos += rng.random(3)[:,None]
    pos[pos>=1.] -= 1.
    pos[pos<0.] += 1.
    pos -= np.mean(pos,axis=1)[:,None] - 0.5
  
  pos -= 0.5 # CM at [0,0,0]
  
  return pos

@njit
def make_positions(BoxSize,GridSize,IDh,glass):

  Nl = GridSize**3 - IDh.size
  Nh = IDh.size * glass.shape[1]
  Ng = glass.shape[1]
  
  pl = np.zeros((Nl,3),dtype=np.float32)
  ph = np.zeros((Nh,3),dtype=np.float32)

  il = 0
  ih = 0

  for cell in range(GridSize**3):
    x = cell // GridSize**2;
    xr = cell % GridSize**2;
    y = xr // GridSize;
    z = xr % GridSize;
    if cell == IDh[ih]:
      if ih*Ng>=Nh:
        raise Exception('high-res array overrun')
      ph[ih*Ng:(ih+1)*Ng,0] = glass[0] + x
      ph[ih*Ng:(ih+1)*Ng,1] = glass[1] + y
      ph[ih*Ng:(ih+1)*Ng,2] = glass[2] + z
      ih += 1
    else:
      if il>=Nl:
        raise Exception('low-res array overrun')
      pl[il,0] = x
      pl[il,1] = y
      pl[il,2] = z
      il += 1

  for i in range(Nh):
    for j in range(3):
      if ph[i,j] < 0.:
        ph[i,j] += GridSize
      if ph[i,j] >= GridSize:
        ph[i,j] -= GridSize

  pl *= BoxSize/GridSize
  ph *= BoxSize/GridSize

  return pl,ph

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <ID-file> <glass-file> <any snapshot> [out-prefix]')
    return 1
  
  header, = gadget_to_particles(argv[3],
    opts={'pos':False,'vel':False,'ID':False,'mass':False})

  num = int(header['NumPart_Total'][1])
  GridSize = int(np.round(num**(1./3)))
  BoxSize = header['BoxSize']

  outprefix = 'init'
  if len(argv) > 4:
    outprefix = argv[4]

  IDh = np.fromfile(argv[1],dtype=np.uint32)
  IDh.sort()
  IDh -= 1 # start from 0

  glass = read_glass(argv[2])
  
  Nl = num - IDh.size
  Nh = IDh.size * glass.shape[1]

  Ml = header['MassTable'][1]
  Mh = Ml / glass.shape[1]

  print('%d particles -> %d low res, %d high res'%(num,Nl,Nh))
  print('low res mass = %e, high res mass = %e'%(Ml,Mh))
  pl,ph = make_positions(BoxSize,GridSize,IDh,glass)

  with h5py.File(outprefix+'.hdf5', 'w') as f:
    print('saving to '+outprefix+'.hdf5')
    header = f.create_group('Header')
    header.attrs['NumPart_ThisFile'] = np.array([0,Nl,Nh,0,0,0],dtype=np.uint32)
    header.attrs['NumPart_Total'] = np.array([0,Nl,Nh,0,0,0],dtype=np.uint64)
    header.attrs['MassTable'] = np.array([0,Ml,Mh,0,0,0],dtype=np.float64)
    header.attrs['Time'] = np.array(0,dtype=np.float64) # GADGET doesn't read this
    header.attrs['Redshift'] = np.array(0,dtype=np.float64) # or this
    header.attrs['BoxSize'] = np.array(BoxSize,dtype=np.float64) # or this
    header.attrs['NumFilesPerSnapshot'] = np.array(1,dtype=np.int32)

    part1 = f.create_group('PartType1')
    part1.create_dataset('Coordinates', data=pl)
    part1.create_dataset('Velocities',  data=np.zeros((Nl,3),dtype=np.float32))
    part1.create_dataset('ParticleIDs', data=np.arange(Nl,dtype=np.uint32))

    part2 = f.create_group('PartType2')
    part2.create_dataset('Coordinates', data=ph)
    part2.create_dataset('Velocities',  data=np.zeros((Nh,3),dtype=np.float32))
    part2.create_dataset('ParticleIDs', data=np.arange(GridSize**3,GridSize**3+Nh,dtype=np.uint32))

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

