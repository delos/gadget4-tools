import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from snapshot_functions import subhalo_group_data

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <group-file> <count> [subhalos=False]')
    return 1

  count = int(argv[2])

  sub = False
  if len(argv) > 3:
    if argv[3].lower() in ['false','f',0]:
      sub = False
    elif argv[3].lower() in ['true','t',1]:
      sub = True
    else:
      raise Exception('unknown boolean')
  
  # read halos
  grp, rank, parentrank, mass, groupmass, header = subhalo_group_data(argv[1])

  num = np.arange(len(rank))

  if not sub:
    idx = rank == parentrank
    print('%d field halos out of %d halos'%(np.sum(idx),np.size(idx)))

    grp = grp[idx]
    num = num[idx]
    mass = groupmass[idx]

  # sort halos by mass
  sort = np.argsort(mass)[::-1]
  grp = grp[sort]
  num = num[sort]
  mass = mass[sort]

  print('# group subhalo mass')
  for i in range(count):
    print('%6d %6d   %.3e'%(grp[i],num[i],mass[i]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
