import numpy as np
from numba import njit
from snapshot_functions import subhalo_desc, read_subhalos, fileprefix_subhalo, particles_by_ID, fileprefix_snapshot

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
  
  if len(argv) < 2:
    print('python script.py <ss,grp>')
    return 1

  use_mostbound = True
  if len(argv) > 2:
    if argv[2].lower() in ['false','f','0']:
      use_mostbound = False
    elif argv[2].lower() in ['true','t','1']:
      use_mostbound = True
    else:
      raise Exception('unknown boolean')

  try: outfile = argv[3]
  except: outfile = 'trace_subhalos.npz'
  
  ss_num, halo_num = np.load(argv[1]).T
  N = halo_num.size

  ssmin = np.min(ss_num[ss_num>=0])
  subs = []
  groups = []
  times = []
  poss = []

  desc = np.full(N,-1)
  mostbound = np.full(N,-1)
  ss = ssmin
  while True:
    # read groups
    try:
      _pos, _sublen, _group, _mostbound, _grplen, _grpfirstsub, _grpnumsubs, header = read_subhalos(
        fileprefix_subhalo%(ss,ss),
        opts={'pos':True,'lentype':True,'group':True,'mostbound':True},
        group_opts={'lentype':True,'firstsub':True,'numsubs':True},
        )
    except Exception as e:
      print(e)
      break

    # add newly formed subhalos
    idx = ss_num==ss
    desc[idx] = halo_num[idx]
    idx = desc>=0

    # connect subhalos to groups
    group = np.full(N,-1)
    group[idx] = _group[desc[idx]]

    # positions
    pos = np.full((N,3),np.nan,dtype=np.float32)

    # recover inactive subhalos
    if use_mostbound:
      mostbound[idx] = _mostbound[desc[idx]]
      idx_ = (~idx) & (mostbound >= 0) # len = len(idx)
      if np.sum(idx_) > 0:
        # get positional data for most bound particles, for inactive subhalos
        pos_, index_, type_, header = particles_by_ID(fileprefix_snapshot%(ss,ss),mostbound[idx_], opts={'pos':True,'index':True,'type':True})
        pos[idx_] = pos_
        # get group data for these particles
        group_ = np.full(np.sum(idx_),-1)
        subhalo_ = np.full(np.sum(idx_),-1)
        for typ in np.unique(type_):
          idxt = type_==typ
          group_[idxt], subhalo_[idxt] = get_groups_subhalos(index_[idxt], _grplen[:,typ], _grpfirstsub, _grpnumsubs, _sublen[:,typ])
        group[idx_] = group_
        desc[idx_] = subhalo_
        idx = desc>=0

    # positions of active subhalos
    pos[idx] = _pos[desc[idx]]

    # record
    times += [header['Time']]
    poss += [pos]
    groups += [group]
    subs += [desc]

    # get subhalo descendants in next snapshot
    desc = np.full(N,-1)
    try:
      desc[idx] = subhalo_desc(ss,subs[-1][idx])[0]
    except Exception as e:
      print(e)
      ss += 1
      break
    ss += 1

  # save output
  np.savez(outfile,snapshot=np.arange(ssmin,ss),time=np.array(times),
    subhalo=np.array(subs),group=np.array(groups),pos=np.array(poss),)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
