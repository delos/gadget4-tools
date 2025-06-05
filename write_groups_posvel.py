import numpy as np
from snapshot_functions import read_subhalos

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <group-file> [output-file]')
    return 1
  
  try: outfile = argv[2]
  except: outfile = 'groupposvel.txt'

  # read halos
  pos, vel, header = read_subhalos(argv[1],opts={},group_opts={'pos':True,'vel':True})
  NG = pos.shape[0]

  print('%d groups'%NG)

  if outfile.endswith('.npz'):
    np.savez(outfile,pos=pos,vel=vel,a=header['Time'])
  else:
    np.savetxt(outfile,np.concatenate((pos,vel),axis=1),header='%.7e'%header['Time'])

if __name__ == '__main__':
  from sys import argv
  run(argv)
