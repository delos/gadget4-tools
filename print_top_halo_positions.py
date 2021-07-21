import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from snapshot_functions import subhalo_group_data

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <group-file> <count> [subhalos=False] [type=None]')
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

  try:
    parttype = int(argv[4])
    print('type %d only'%parttype)
  except:
    parttype = None
  
  # read halos
  grp, rank, parentrank, mass, groupmass, length, grouplength, pos, grouppos, header = subhalo_group_data(
    argv[1],parttype=parttype,opts={'mass':True,'len':True,'pos':True})

  num = np.arange(len(rank))

  if not sub:
    idx = rank == parentrank
    print('%d field halos out of %d halos'%(np.sum(idx),np.size(idx)))

    grp = grp[idx]
    num = num[idx]
    mass = groupmass[idx]
    length = grouplength[idx]
    pos = grouppos[idx]

  # sort halos by mass
  sort = np.argsort(mass)[::-1]
  sort = np.argsort(length)[::-1]
  grp = grp[sort]
  num = num[sort]
  mass = mass[sort]
  length = length[sort]
  pos = pos[sort]

  BoxSize = header['BoxSize']

  if count <= 0:
    count = num.size
  else:
    count = min(count,num.size)

  print('# group subhalo   mass      x/box    y/box    z/box  length')
  for i in range(count):
    print('%6d %6d   %.3e %.6f %.6f %.6f  %d'%(grp[i],num[i],mass[i],
      pos[i,0]/BoxSize,pos[i,1]/BoxSize,pos[i,2]/BoxSize,length[i]))

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
