import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

def group_data(fileprefix,opts={'mass':True,'rad':False,'len':False,'pos':False}):

  '''
  
  Read halos from group file and return data.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_subhalo_tab_000, not fof_subhalo_tab_000.0.hdf5)

    opts: which fields to read and return

  Returns:
    
    mass: SO mass M_200m

    rad: SO radius R_200m

    length: group particle count

    pos: position (NH,3)

    header: a dict with header info, use list(header) to see the fields
    
  '''

  filepath = [
    Path(fileprefix + '.hdf5'),
    Path(fileprefix + '.0.hdf5'),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d.hdf5'
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1

  fileinst = 0
  if opts.get('mass'):
    mass = []
  if opts.get('rad'):
    rad = []
  if opts.get('len'):
    length = []
  if opts.get('pos'):
    pos = []
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      ScaleFactor = 1./(1+header['Redshift'])
      numfiles = header['NumFiles']

      if opts.get('mass'):
        mass += [np.array(f['Group/Group_M_Mean200'])]
      if opts.get('rad'):
        rad += [np.array(f['Group/Group_R_Mean200'])]
      if opts.get('len'):
        length += [np.array(f['Group/GroupLen'])]
      if opts.get('pos'):
        pos += [np.array(f['Group/GroupPos'])]

    fileinst += 1

  ret = []

  if opts.get('mass'):
    mass = np.concatenate(mass)
    ret += [mass]
  if opts.get('rad'):
    rad = np.concatenate(rad)
    ret += [rad]
  if opts.get('len'):
    length = np.concatenate(length)
    ret += [length]
  if opts.get('pos'):
    pos = np.concatenate(pos,axis=0)
    ret += [pos]

  return tuple(ret + [header])

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <group-file> <count>')
    return 1

  count = int(argv[2])

  # read halos
  mass,rad,length,pos,header = group_data(argv[1],opts={'mass':True,'rad':True,'len':True,'pos':True})

  num = np.arange(len(mass))

  # sort halos by mass
  sort = np.argsort(mass)[::-1]
  mass = mass[sort]
  rad = rad[sort]
  length = length[sort]
  pos = pos[sort]

  BoxSize = header['BoxSize']

  if count <= 0:
    count = num.size
  else:
    count = min(count,num.size)

  boxdig = int(np.log10(BoxSize)+.5)
  if 0 <= boxdig < 8:
    fmt = '%9.' + '%df'%(7-boxdig)
  else:
    fmt = '%.6e'

  print('# BoxSize = ' + fmt%BoxSize)
  print('# number, mass, x, y, z, r, length')
  for i in range(count):
    print('%6d %.3e '%(num[i],mass[i]) + ' '.join(fmt%x for x in pos[i]) + ' ' + fmt%rad[i] + ' %9d'%length[i])

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
