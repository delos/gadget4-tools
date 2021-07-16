import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from snapshot_functions import subhalo_group_data

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <group-file> <mass> [include subhalos=False]')
    return 1

  masspick = float(argv[2])

  sub = False
  if len(argv) > 3:
    if argv[3].lower() in ['false','f',0]:
      sub = False
    elif argv[3].lower() in ['true','t',1]:
      sub = True
    else:
      raise Exception('unknown boolean')
  
  # read halos
  grp, rank, parentrank, mass, header = subhalo_group_data(argv[1])

  num = np.arange(len(rank))

  if not sub:
    idx = rank == parentrank
    print('%d field halos out of %d halos'%(np.sum(idx),np.size(idx)))

    grp = grp[idx]
    num = num[idx]
    mass = mass[idx]

  # sort halos by mass
  sort = np.argsort(mass)
  grp = grp[sort]
  num = num[sort]
  mass = mass[sort]

  idx = np.where(mass<masspick)[0][-1]
  if idx < len(mass)-1 and mass[idx+1]/masspick > masspick/mass[idx]:
    idx += 1

  print('mass = %g'%mass[idx])
  print('subhalo number = %d'%num[idx])
  print('group number = %d'%grp[idx])

  ax = plt.figure().gca()

  ax.loglog(mass[::-1],np.arange(len(mass))+1)
  ax.axvline(mass[idx])

  ax.set_xlabel(r'$M$ ($M_\odot/h$)')
  ax.set_ylabel(r'$N(>M)$')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
