import numpy as np
from numba import njit
from snapshot_functions import group_extent, read_particles_filter, fileprefix_snapshot, fileprefix_subhalo

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot min,max> <trace-file> <peak/ID file> [out-file]')
    return 1

  ssmin,ssmax = [int(x) for x in argv[1].split(',')]

  _data = np.loadtxt(argv[2])
  _ss = _data[:,0]
  _x = _data[:,2]
  _y = _data[:,3]
  _z = _data[:,4]
  _grp = _data[:,10].astype(int)
  print('using trace file')

  if ssmin < _ss[0]:
    ssmin = _ss[0]
  if ssmax > _ss[-1] or ssmax < 0:
    ssmax = _ss[-1]
  print('  snapshots %d-%d'%(ssmin,ssmax))

  outfile = 'peaks_in_halo.txt'
  if len(argv) > 4:
    outfile = argv[4]

  peakdata = np.loadtxt(argv[3]).astype(int)

  peakn = peakdata[:,0]
  peakID = peakdata[:,1:]

  IDs = np.unique(peakID)

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

