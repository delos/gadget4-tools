import numpy as np
from snapshot_functions import subhalo_desc, read_subhalos, fileprefix_subhalo

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <ss_halo.npy> [out-file.npz]')
    return 1

  try: outfile = argv[2]
  except: outfile = 'trace_subhalos.npz'
  
  ss_num, halo_num = np.load(argv[1]).T
  N = halo_num.size

  ssmin = np.min(ss_num[ss_num>=0])
  subs = []
  groups = []
  times = []

  desc = np.full(N,-1)
  ss = ssmin
  while True:
    # read groups
    try:
      _group, header = read_subhalos(fileprefix_subhalo%(ss,ss),opts={'group':True})
    except:
      break

    # add newly formed subhalos and store
    idx = ss_num==ss
    desc[idx] = halo_num[idx]
    subs += [desc]
    idx = subs[-1]>=0

    # connect subhalos to groups, store groups to output
    group = np.full(N,-1)
    group[idx] = _group[subs[-1][idx]]
    groups += [group]

    # store time to output
    times += [header['Time']]

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
    subhalo=np.array(subs).T,group=np.array(groups).T)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
