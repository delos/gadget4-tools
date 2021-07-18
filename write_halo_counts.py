import numpy as np
import matplotlib.pyplot as plt
from snapshot_functions import count_halos

def run(argv):
  
  if len(argv) < 1:
    print('python script.py')
    return 1

  num, time, groups, subhalos, mass = count_halos()

  with open('halos.txt','wt') as f:
    f.write('# snapshot, a, groups, subhalos, mass\n')
    for i in range(len(num)):
      f.write('%d %.6e %d %d %.6e\n'%(
        num[i], time[i], groups[i], subhalos[i], mass[i]))
  
  ax = plt.figure().gca()

  ax.loglog(time,mass)

  ax.set_ylabel(r'$M$ ($M_\odot/h$)')
  ax.set_xlabel(r'$a$')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
