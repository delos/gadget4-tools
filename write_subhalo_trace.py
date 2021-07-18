import numpy as np
import matplotlib.pyplot as plt
from snapshot_functions import trace_subhalo

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot> <halo>')
    return 1
  
  ss_num = int(argv[1])
  halo_num = int(argv[2])

  num, time, pos, vel, mass, ID = trace_subhalo(ss_num,halo_num)

  with open('trace_%d_%d.txt'%(ss_num,halo_num),'wt') as f:
    f.write('# snapshot, a, x, y, z, vx, vy, vz, groupmass, subhalomass, group, subhalo, particle\n')
    for i in range(len(num)):
      f.write('%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %d %d %d\n'%(
        num[i], time[i], pos[i,0], pos[i,1], pos[i,2], vel[i,0], vel[i,1], vel[i,2],
        mass[i]['group'], mass[i]['subhalo'], ID[i]['group'], ID[i]['subhalo'], ID[i]['particle']))
  
  ax = plt.figure().gca()

  ax.loglog(time,[m['subhalo'] for m in mass],'r',label='subhalo')
  ax.loglog(time,[m['group'] for m in mass],'k--',label='group')

  ax.set_ylabel(r'$M$ ($M_\odot/h$)')
  ax.set_xlabel(r'$a$')
  ax.legend()
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
