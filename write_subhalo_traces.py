import numpy as np
from snapshot_functions import subhalo_desc, read_subhalos, fileprefix_subhalo, particles_by_ID, fileprefix_snapshot

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <ss_halo.npy> [use most bound=1] [out-file.npz]')
    return 1

  use_mostbound = True
  if len(argv) > 2:
    if argv[2].lower() in ['false','f',0]:
      use_mostbound = False
    elif argv[2].lower() in ['true','t',1]:
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
      _pos, _group, _mostbound, header = read_subhalos(fileprefix_subhalo%(ss,ss),opts={'pos':True,'group':True,'mostbound':True})
    except:
      break

    # add newly formed subhalos and store
    idx = ss_num==ss
    desc[idx] = halo_num[idx]
    subs += [desc]

    # indices of active subhalos
    idx = subs[-1]>=0

    # connect subhalos to groups, store groups to output
    group = np.full(N,-1)
    group[idx] = _group[subs[-1][idx]]
    groups += [group]

    # store time to output
    times += [header['Time']]

    # positions of active subhalos
    pos = np.full((N,3),np.nan,dtype=np.float32)
    pos[idx] = _pos[subs[-1][idx]]

    if use_mostbound:
			# positions of inactive subhalos with most bound IDs
      mostbound[idx] = _mostbound[subs[-1][idx]]
			idx_ = (~idx) & (mostbound >= 0)
			if np.sum(idx_) > 0:
				pos_, header = particles_by_ID(fileprefix_snapshot%(ss,ss),mostbound[idx_],opts={'pos':True})
				pos[idx_] = pos_
    poss += [pos]

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
