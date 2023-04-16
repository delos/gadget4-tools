import numpy as np
from snapshot_functions import read_subhalos

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <group-file> [output-file]')
    return 1
  
  try: outfile = argv[2]
  except: outfile = 'subhalopos.npz'

  pos, mass, rad, grp, rank, par, header = read_subhalos(argv[1],
    opts={'pos':True,'mass':True,'radius':True,'group':True,'rank':True,'parentrank':True})
  sub = np.arange(mass.size)

  print('%d subhalos'%(mass.size))

  np.savez(outfile,a=header['Time'],mass=mass,pos=pos,rad=rad,group=grp,parent=sub+par-rank)

if __name__ == '__main__':
  from sys import argv
  run(argv)
