import numpy as np
from snapshot_functions import read_particles_filter

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <data-file> <preIC-file> [rotation-file=0] [out-file]')
    return 1

  datafile = argv[1]
  prefile = argv[2]

  rotation = None
  if len(argv) > 3:
    try:
      rotation = np.loadtxt(argv[3])
      print('rotation matrix:')
      with np.printoptions(precision=3, suppress=True):
        print(rotation)
    except:
      rotation = None
      print('no rotation')

  outfile = 'Lagrangian_field.npz'
  if len(argv) > 4:
    outfile = argv[4]

  data = np.load(datafile)

  ID_ = data['ID']
  val_ = data[list(data)[list(data) != 'ID'][0]]

  sort_ = np.argsort(ID_)
  ID_ = ID_[sort_]
  val_ = val_[sort_]

  print('%d particles in data'%len(ID_))

  pos, ID, header = read_particles_filter(prefile,rotation=None,ID_list=ID_,opts={'pos':True,'ID':True})

  sort = np.argsort(ID)
  ID = ID[sort]
  pos = pos[sort]

  print('%d particles in ICs'%len(ID))

  for i in range(3):
    pos -= np.mean(pos,axis=0)[None,:]
    pos[pos >= 0.5*header['BoxSize']] -= header['BoxSize']
    pos[pos < -0.5*header['BoxSize']] += header['BoxSize']
  if rotation is not None:
    pos = (rotation@(pos.T)).T

  np.savez(outfile,pos=pos.astype(np.float32),val=val_.astype(np.float32))

  radius = np.sqrt(np.sum(pos**2,axis=1).max())
  print('radius = %g'%radius)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

