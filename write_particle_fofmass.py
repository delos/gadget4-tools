import numpy as np
from snapshot_functions import read_subhalos, read_particles_filter, fileprefix_snapshot, fileprefix_subhalo, file_extension
from pathlib import Path

def grppartID(ss,typ,bins,length):
  ID, header = read_particles_filter(fileprefix_snapshot%(ss,ss),type_list=[typ],opts={'ID':True},part_range=(bins[0],bins[-1]))
  ix = np.full(bins[-1,typ],False)
  for i in range(len(bins[:,typ])-1):
    ix[bins[i,typ]:bins[i,typ]+sublen[i,typ]] = True
  return ID[ix]

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
  M, Length, header = read_subhalos(prefix_sub,opts={},group_opts={'mass':True,'lentype':True})
  NG = M.size
  NP = np.sum(Length,axis=0)
  print('%d groups, '%NG + ','.join(['%d'%N for N in NP]) + ' particles')

  partM = np.concatenate([np.concatenate([np.full(Length[i,typ],M[i],dtype=np.float32) for i in range(NG)]) for typ in range(6)],dtype=np.float32)
  partID = np.concatenate([read_particles_filter(prefix,type_list=[typ],opts={'ID':True},part_range=(np.zeros_like(NP),NP))[0] for typ in range(6)],dtype=np.uint32)
  sort = np.argsort(partID)

  np.savez('particles_fofmass_%03d.npz'%ss,ID=partID[sort],M=partM[sort])

if __name__ == '__main__':
  from sys import argv
  run(argv)
