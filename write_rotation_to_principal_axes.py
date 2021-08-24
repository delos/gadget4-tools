import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter
from scipy.linalg import eigh

def run(argv):
  
  if len(argv) < 5:
    print('python script.py <IC-file> <preIC-file> <ID> <radius>')
    return 1

  ID = int(argv[3])
  r = float(argv[4])

  print('getting IDs of nearby particles')
  pos, header = read_particles_filter(argv[2],ID_list=[ID],opts={'pos':True})
  IDs, header = read_particles_filter(argv[2],center=pos[0],radius=r,opts={'ID':True})

  print('reading positions of %d particles'%len(IDs))
  pos0, ID0, header = read_particles_filter(argv[2],ID_list=IDs,opts={'pos':True,'ID':True})
  sort0 = np.argsort(ID0)
  ID0 = ID0[sort0]
  pos0 = pos0[sort0] - pos

  pos1, ID1, header = read_particles_filter(argv[1],ID_list=IDs,opts={'pos':True,'ID':True})
  sort1 = np.argsort(ID1)
  ID1 = ID1[sort1]
  pos1 = pos1[sort1] - pos

  if not np.array_equal(ID0,ID1):
    print('Error')
    print(np.stack((ID0,ID1)).T.tolist())
    return

  rot = np.diag((1,1,1))

  for i in range(2):
    if i > 0:
      eigval, eigvec = eigh(e)
      rot1 = eigvec.T

      print('rotate by %.0f degrees'%(np.arccos((np.trace(rot1)-1)/2)*180./np.pi))

      pos = (rot1 @ (pos.T)).T
      pos0 = (rot1 @ (pos0.T)).T
      pos1 = (rot1 @ (pos1.T)).T

      rot = rot1 @ rot

    disp = pos1 - pos0
    
    e = np.zeros((3,3))
    for c in range(3):
      dist2 = np.zeros(pos0.shape[0])
      for d in range(3):
        if d != c: dist2 += pos0[:,d]**2
      idx = np.argsort(dist2)[:32]
      for d in range(3):
        e[c,d] = np.polyfit(pos0[idx,c],disp[idx,d],1)[0]

    e = .5*(e + e.T)
    with np.printoptions(precision=5, suppress=True):
      print('Tidal tensor:')
      print(e)

  with np.printoptions(precision=5, suppress=True):
    print('rotation matrix (%.0f degrees)'%(np.arccos((np.trace(rot)-1)/2)*180./np.pi))
    print(rot)

  np.savetxt('rotation_%d.txt'%ID,rot)
  
if __name__ == '__main__':
  from sys import argv
  run(argv)                         
