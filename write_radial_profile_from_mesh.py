import numpy as np
import os
from sys import argv

# to do: optimize this
def sphericalAverage(image, center=None):
    """
    Calculate the spherically-averaged radial profile.

    image - The 3D image
    center - The [x,y,z] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    z, y, x = np.indices(image.shape)
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0,
                           (z.max()-z.min())/2.0])

    r = np.sqrt( (x - center[0])**2 + 
                 (y - center[1])**2 + 
                 (z - center[2])**2 )

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def run(argv):

  if len(argv) < 2:
    print('python script.py <filename>')
    return 1

  filename = argv[1]

  dsize = os.stat(filename).st_size//4
  GridSize = int(dsize**(1./3)+.5)
  print('n = %d'%GridSize)
  with open(filename,'rb') as f:
    field = np.fromfile(f,count=-1,dtype=np.float32)
  field.shape = (GridSize,GridSize,GridSize)

  prof = sphericalAverage(field)

  np.savetxt('profile.txt',np.stack(((np.arange(len(prof))+.5)/GridSize,prof)).T)

if __name__ == '__main__':
  from sys import argv
  run(argv)
