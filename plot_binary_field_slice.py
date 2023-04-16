import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = cm.viridis

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <filename> <axis,min,max> [log=1,positive=0,square=0] [x min,max] [y min,max]')
    return 1
  
  filename = argv[1]
  
  axis = argv[2].split(',')
  amin = int(axis[1])
  amax = int(axis[2])
  axis = int(axis[0])

  log = 1
  positive = 0
  square = 0
  if len(argv) > 3:
    logsq = argv[3].split(',')
    log = int(logsq[0])
    if len(logsq) > 1:
      positive = int(logsq[1])
    if len(logsq) > 2:
      square = int(logsq[2])

  try: xlim = np.array(argv[4].split(','),dtype=int)
  except: xlim = None
  try: ylim = np.array(argv[5].split(','),dtype=int)
  except: ylim = None

  dsize = os.stat(filename).st_size//4
  GridSize = int(dsize**(1./3)+.5)

  if axis == 0:
    only_read = [amin,amax]
  elif xlim is not None:
    only_read = xlim
  else:
    only_read = [0,GridSize-1]

  with open(filename,'rb') as f:
    f.seek(GridSize**2*only_read[0]*4)
    delta = np.fromfile(f,count=GridSize**2*(only_read[1]-only_read[0]+1),dtype=np.float32)
  delta.shape = (only_read[1]-only_read[0]+1,GridSize,GridSize)
  bins = np.arange(GridSize+1)

  if square:
    if not positive:
      delta += 1.
    delta *= delta

  if axis == 0:
    delta = np.mean(delta,axis=0)
    if xlim is not None:
      delta = delta[xlim[0]:xlim[1]+1]
  elif axis == 1:
    delta = np.mean(delta[:,amin:amax+1],axis=axis)
  if axis == 2:
    delta = np.mean(delta[:,:,amin:amax+1],axis=axis)

  if ylim is not None:
    delta = delta[:,ylim[0]:ylim[1]+1]

  if square:
    delta = np.sqrt(delta)
    if not positive:
      delta -= 1.

  if log:
    if not positive:
      np.log10(1+delta,out=delta)
    else:
      np.log10(delta,out=delta)
  
  fig, ax = plt.subplots(figsize=(6., 5))
  im = ax.imshow(delta.T,origin='lower')
  cbar = fig.colorbar(im, ax=ax)
  ax.patch.set_facecolor('k')
  
  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
