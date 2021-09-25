import numpy as np
import matplotlib.pyplot as plt

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <table> [rho or m or n] [scale]')
    return 1

  which = 'rho'
  if len(argv) > 2:
    if argv[2].lower() in ['rho','density','dens']:
      which = 'rho'
    elif argv[2].lower() in ['m','mass']:
      which = 'mass'
    elif argv[2].lower() in ['n','num','number','ct','count','par','particles']:
      which = 'number'

  try: scale = float(argv[3])
  except: scale = 0.
  
  r, rho, ru, m, N = np.loadtxt(argv[1]).T
  
  ax = plt.figure().gca()
  
  ax.set_xlim(r[0],ru[-1])
  ax.set_xlabel(r'$r$')

  if which == 'rho':
    ax.loglog(r,rho*r**scale)
    if scale != 0.:
      ax.set_ylabel('$\\rho r^{%.1f}$'%scale)
    else:
      ax.set_ylabel(r'$\rho$')
  elif which == 'mass':
    ax.loglog(ru,m*ru**scale)
    if scale != 0.:
      ax.set_ylabel('$m r^{%.1f}$'%scale)
    else:
      ax.set_ylabel(r'$m$')
  elif which == 'number':
    ax.loglog(ru,N*ru**scale)
    if scale != 0.:
      ax.set_ylabel('$N r^{%.1f}$'%scale)
    else:
      ax.set_ylabel(r'$N$')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
