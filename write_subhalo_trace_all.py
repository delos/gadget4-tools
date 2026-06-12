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
    print('python script.py [use most bound=1] [out-file.npz]')
    return 1

  use_mostbound = True
  if len(argv) > 1:
    if argv[1].lower() in ['false','f','0']:
      use_mostbound = False
    elif argv[1].lower() in ['true','t','1']:
      use_mostbound = True
    else:
      raise Exception('unknown boolean')

  try: outfile = argv[2]
  except: outfile = 'trace_subhalos.npz'
  
  subs = []
  groups = []
  times = []
  snaps = []
  poss = []
  vels = []
  masss = []

  N = 0
  desc = np.full(N,-1)
  mostbound = np.full(N,-1)
  ss = 0
  while True:
    # read groups
    try:
      _pos, _vel, _mass, _sublen, _group, _mostbound, _grplen, _grpfirstsub, _grpnumsubs, header = read_subhalos(
        fileprefix_subhalo%(ss,ss),
        opts={'pos':True,'vel':True,'mass':True,'lentype':True,'group':True,'mostbound':True},
        group_opts={'lentype':True,'firstsub':True,'numsubs':True},
        )
    except Exception as e:
      print(e)
      ss += 1
      continue

    # add newly formed subhalos
    Nsub = _pos.shape[0]
    newsub = np.arange(Nsub)
    newsub = newsub[~np.isin(newsub,desc)]
    sub = np.concatenate((desc,newsub))
    mostbound = np.concatenate((mostbound,np.full(newsub.size,-1)))
    N = sub.size
    idx = sub>=0
    print('%d subhalos, %d active'%(N,np.sum(idx)))

    # connect subhalos to groups
    group = np.full(N,-1)
    group[idx] = _group[sub[idx]]

    # positions
    pos = np.full((N,3),np.nan,dtype=np.float32)
    vel = np.full((N,3),np.nan,dtype=np.float32)
    mass = np.full(N,np.nan,dtype=np.float32)

    # recover inactive subhalos
    if use_mostbound:
      mostbound[idx] = _mostbound[sub[idx]]
      idx_ = (~idx) & (mostbound >= 0) # len = len(idx)
      if np.sum(idx_) > 0:
        # get positional data for most bound particles, for inactive subhalos
        pos_, vel_, index_, type_, header = particles_by_ID(fileprefix_snapshot%(ss,ss),mostbound[idx_], opts={'pos':True,'vel':True,'index':True,'type':True})
        pos[idx_] = pos_
        vel[idx_] = vel_
        mass[idx_] = 0.
        # get group data for these particles
        group_ = np.full(np.sum(idx_),-1)
        subhalo_ = np.full(np.sum(idx_),-1)
        for typ in np.unique(type_):
          idxt = type_==typ
          group_[idxt], subhalo_[idxt] = get_groups_subhalos(index_[idxt], _grplen[:,typ], _grpfirstsub, _grpnumsubs, _sublen[:,typ])
        group[idx_] = group_
        sub[idx_] = subhalo_
        idx = sub>=0
    print('  --> %d active'%(np.sum(idx)))

    # positions of active subhalos
    pos[idx] = _pos[sub[idx]]
    vel[idx] = _vel[sub[idx]]
    mass[idx] = _mass[sub[idx]]

    # record
    times += [header['Time']]
    snaps += [ss]
    poss += [pos]
    vels += [vel]
    masss += [mass]
    groups += [group]
    subs += [sub]

    # get subhalo descendants in next snapshot
    desc = np.full(N,-1)
    try:
      desc[idx] = subhalo_desc(ss,sub[idx])[0]
    except Exception as e:
      print(e)
      ss += 1
      break
    ss += 1

  # homogenize
  for i in range(len(snaps)):
    Nadd = N - subs[i].size
    subs[i] = np.concatenate((subs[i],np.full(Nadd,-1)))
    groups[i] = np.concatenate((groups[i],np.full(Nadd,-1)))
    poss[i] = np.concatenate((poss[i],np.full((Nadd,3),np.nan,dtype=np.float32)),axis=0,dtype=np.float32)
    vels[i] = np.concatenate((vels[i],np.full((Nadd,3),np.nan,dtype=np.float32)),axis=0,dtype=np.float32)
    masss[i] = np.concatenate((masss[i],np.full(Nadd,np.nan,dtype=np.float32)),dtype=np.float32)

  # save output
  np.savez(outfile,snapshot=np.array(snaps),time=np.array(times),
    subhalo=np.array(subs),group=np.array(groups),
    pos=np.array(poss),vel=np.array(vels),mass=np.array(masss),)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
