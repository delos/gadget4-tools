import numpy as np
from snapshot_functions import read_subhalos

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <group-file> [output-file]')
    return 1
  
  try: outfile = argv[2]
  except: outfile = 'groupmass.txt'

  # read halos
  mass, somass, header = read_subhalos(argv[1],opts={},
    group_opts={'mass':True,'somass':True},sodef='Mean200')
  NG = mass.size

  print('%d groups'%NG)

  if outfile.endswith('.npz'):
    np.savez(outfile,mass=mass,somass=somass,a=header['Time'])
  else:
    np.savetxt(outfile,np.concatenate((mass[:,None],somass[:,None]),axis=1),header='%.7e'%header['Time'])

if __name__ == '__main__':
  from sys import argv
  run(argv)
