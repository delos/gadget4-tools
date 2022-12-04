import numpy as np
import re
from sys import argv

def run(argv):
  files = np.sort(argv[1:])

  rholist = []
  mlist = []
  Nlist = []

  alist = []
  poslist = []

  for file in files:
    with open(file,'rt') as f:
      lines = f.readlines()
      alist += [float(lines[1][1:])]
      poslist += [[float(x) for x in re.split(' |\(|\)|,|\n',lines[0][1:]) if x]]
    print(file,alist[-1],poslist[-1])
    r, rho, ru, m, N = np.loadtxt(file).T[:5]
    rholist += [rho.astype(np.float32)]
    mlist += [m.astype(np.float32)]
    Nlist += [N.astype(np.int32)]

  np.savez('profiles.npz',r=r,ru=ru,
    rho=np.array(rholist),
    m=np.array(mlist),
    N=np.array(Nlist),
    time=np.array(alist),
    pos=np.array(poslist))

  return

if __name__ == '__main__':
  from sys import argv
  run(argv)
