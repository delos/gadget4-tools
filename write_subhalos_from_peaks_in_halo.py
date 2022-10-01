import numpy as np
from numba import njit
from snapshot_functions import group_extent, read_particles_filter, fileprefix_snapshot, fileprefix_subhalo

def get_peaks(file):
  out = {}
  with open(file) as f:
    for line in f:
      data = (line.rstrip()).split()
      out[int(data[0])] = np.array([int(x) for x in data[1:]])
  return out

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot min,max> <peaks-in-halo> <peak-particles> [out-file]')
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

  print(peaks,IDs)
  raise


  npart = peakID.shape[1]
  print('%d peaks in file; %d IDs per peak'%(peakn.size,npart))

  for i, ss in enumerate(np.arange(ssmin,ssmax+1)):
    grp = _grp[_ss==ss][0]
    print('snapshot %d: group %d'%(ss,grp))

    ssname = fileprefix_snapshot%(ss,ss)
    grpname = fileprefix_subhalo%(ss,ss)
    gpos, grad, header = group_extent(grpname,grp,size_definition='Mean200')

    if grad <= 0:
      continue

    ID, _ = read_particles_filter(ssname,center=gpos,radius=grad,ID_list=IDs,opts={'pos':False,'vel':False,'ID':True,'mass':False})

    if len(ID) == 0:
      continue
    
    inhalo = np.any(np.isin(peakID,ID),axis=1)

    peaksinhalo = peakn[inhalo]

    with open(outfile,'at') as f:
      f.write('%d '%ss)
      for n in peaksinhalo:
        f.write(' %d'%n)
      f.write('\n')

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

