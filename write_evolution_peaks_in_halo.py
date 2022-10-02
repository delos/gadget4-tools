import numpy as np
from numba import njit
from snapshot_functions import read_subhalos, particles_by_ID, fileprefix_snapshot, fileprefix_subhalo

def get_peaks(file):
  out = {}
  with open(file) as f:
    for line in f:
      data = (line.rstrip()).split()
      out[int(data[0])] = np.array([int(x) for x in data[1:]])
  return out

@njit
def get_groups_subhalos(index, group_length, group_firstsub, group_numsubs, subhalo_length):
  group = np.zeros(index.size,dtype=np.int32)-1
  subhalo = np.zeros(index.size,dtype=np.int32)-1
  p = 0
  for i,gl in enumerate(group_length):
    group[(p <= index)&(index < p+gl)] = i
    q = 0
    for j,sl in enumerate(subhalo_length[group_firstsub[i]:group_firstsub[i]+group_numsubs[i]]):
      subhalo[(p+q <= index)&(index < p+q+sl)] = group_firstsub[i] + j
      q += sl
    p += gl
  return group, subhalo

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot min,max> <peaks-in-halo> <peak-particles>')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]
  print('  snapshots %d-%d'%(ssmin,ssmax))

  #peaks = np.unique(np.concatenate(get_peaks(argv[2])))
  peaks_dict = get_peaks(argv[2])
  peaks = np.unique(np.concatenate([peaks_dict[x] for x in list(peaks_dict)]))

  peakdata = np.loadtxt(argv[3]).astype(int)
  peakn = peakdata[:,0]
  peakID = peakdata[:,1:]

  IDs = np.array([peakID[peakn==peak][0] for peak in peaks])

  snapshots = np.arange(ssmin,ssmax+1)
  groups = np.zeros(snapshots.shape+IDs.shape,dtype=np.int32)-1
  subhalos = np.zeros(snapshots.shape+IDs.shape,dtype=np.int32)-1
  pos = np.zeros(snapshots.shape+IDs.shape+(3,),dtype=np.float32)*np.nan
  vel = np.zeros(snapshots.shape+IDs.shape+(3,),dtype=np.float32)*np.nan
  subpos = np.zeros(snapshots.shape+IDs.shape+(3,),dtype=np.float32)*np.nan
  subvel = np.zeros(snapshots.shape+IDs.shape+(3,),dtype=np.float32)*np.nan

  for i, ss in enumerate(np.arange(ssmin,ssmax+1)):
    ssname = fileprefix_snapshot%(ss,ss)
    grpname = fileprefix_subhalo%(ss,ss)

    try:
      _subpos, _subvel, _sublen, _subgrp, _grplen, _grpfirstsub, _grpnumsubs, _ = read_subhalos(grpname,opts={'pos':True,'vel':True,'lentype':True,'group':True},group_opts={'lentype':True,'firstsub':True,'numsubs':True})
    except Exception as e:
      print(e)
      continue
    _pos, _vel, _index, _type, header = particles_by_ID(ssname, IDs, opts={'pos':True,'vel':True,'index':True,'type':True})

    for typ in np.unique(_type):
      idx = (_type==typ)
      group, subhalo = get_groups_subhalos(_index[idx], _grplen[:,typ], _grpfirstsub, _grpnumsubs, _sublen[:,typ])
      groups[i,idx] = group
      subhalos[i,idx] = subhalo
      subpos[i,idx] = _subpos[subhalo]
      subvel[i,idx] = _subvel[subhalo]
    pos[i] = _pos
    vel[i] = _vel

  np.savez('peak_evolution.npz',snapshots=snapshots, peaks=peaks, IDs=IDs, groups=groups, subhalos=subhalos, pos=pos, vel=vel, subpos=subpos, subvel=subvel)

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

