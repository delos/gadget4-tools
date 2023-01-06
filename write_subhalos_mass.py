import numpy as np
from snapshot_functions import read_subhalos

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <group-file> [positions=0] [output-file]')
    return 1
  
  try: outfile = argv[3]
  except: outfile = 'subhalomass.txt'

  if len(argv) > 2 and not argv[2].lower().startswith(('f','0','n')):
    pos, mass, header = read_subhalos(argv[1],opts={'pos':True,'mass':True,})
  else:
    mass, header = read_subhalos(argv[1],opts={'mass':True,})
    pos = None

  print('%d subhalos'%(mass.size))

  if outfile.endswith('.npz'):
    np.savez(outfile,mass=mass,a=header['Time'],pos=pos)
  else:
    if pos is None:
      np.savetxt(outfile,mass,header='%.7e'%header['Time'])
    else:
      np.savetxt(outfile,np.concatenate((mass[:,None],pos),axis=1),header='%.7e'%header['Time'])

if __name__ == '__main__':
  from sys import argv
  run(argv)
