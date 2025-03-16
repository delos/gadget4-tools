import numpy as np
from snapshot_functions import read_subhalos, read_particles_filter, fileprefix_snapshot, fileprefix_subhalo, file_extension
from pathlib import Path

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <ss>')
    return 1
  
  ss = int(argv[1])

  prefix = fileprefix_snapshot%(ss,ss)
  prefix_sub = fileprefix_subhalo%(ss,ss)
  filepath = [
    Path(prefix + file_extension),
    Path(prefix + '.0' + file_extension),
    Path(prefix.split('/')[-1] + file_extension),
    ]

  if filepath[0].is_file():
    numfiles = 1
  elif filepath[1].is_file():
    numfiles = 2
  elif filepath[2].is_file():
    prefix = prefix.split('/')[-1]
    prefix_sub = prefix_sub.split('/')[-1]
    numfiles = 1

  # read halos
  X, M, R, header = read_subhalos(prefix_sub,opts={},
    group_opts={'pos':True,'somass':True,'soradius':True,},sodef='Mean200')
  NG = M.size

  print('%d groups'%NG)

  #ID, header = read_particles_filter(prefix,opts={'ID':True,})
  #print(ID.min(),ID.max())

  for i in range(NG):
    print('%d/%d'%((i+1),NG))

    #m, ID, header = read_particles_filter(prefix,center=X[i],radius=R[i],opts={'ID':True,'mass':True})
    #print(i,np.size(m),np.sum(m),M[i])

    ID, header = read_particles_filter(prefix,center=X[i],radius=R[i],opts={'ID':True,})

    if i == 0:
      NP = header['NumPart_Total'][1]
      Mmax = np.zeros(NP)
      Mmin = np.full(NP,np.inf)
    Mmax[ID-1] = np.maximum(Mmax[ID-1],M[i])
    Mmin[ID-1] = np.minimum(Mmin[ID-1],M[i])

  Mmax[Mmax==0.] = np.nan
  Mmin[np.isinf(Mmin)] = np.nan

  ID = np.arange(1,NP+1)
  ix = np.isfinite(Mmax)&np.isfinite(Mmin)

  np.savez('particles_somass_%d.npz'%ss,ID=ID[ix].astype(np.uint32),Mmax=Mmax[ix].astype(np.float32),Mmin=Mmin[ix].astype(np.float32))

if __name__ == '__main__':
  from sys import argv
  run(argv)
