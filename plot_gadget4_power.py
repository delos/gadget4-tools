import numpy as np
import matplotlib.pyplot as plt

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <table>')
    return 1
  
  with open(argv[1],'rt') as f:
    lines = f.readlines()
    Time = float(lines[0])
    Bins = int(lines[1])
    BoxSize = float(lines[2])
    Ngrid = int(lines[3])
    Dgrowth = float(lines[4])

    k = np.zeros(Bins)
    P = np.zeros(Bins)

    for i in range(Bins):
      kP = lines[i+5].split()
      k[i] = kP[0]
      P[i] = kP[1]
  
  ax = plt.figure().gca()
  
  ax.loglog(k,P)
  ax.set_xlim(k[0],k[-1])
  ax.set_xlabel(r'$k (h/kpc)$')
  ax.set_ylabel(r'dimensionless $\mathcal{P}(k)$')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
