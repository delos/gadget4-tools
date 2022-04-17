import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy.integrate import quad
from scipy.spatial import cKDTree as KDTree
#from scipy.spatial import KDTree
from numba import njit

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

@njit
def dtda(a,h,OM):
  OL = 1. - OM
  H = 0.1*h*np.sqrt(OM*a**-3+OL) # = da/dt / a
  return 1./(a*H)

def run(argv):
  
  if len(argv) < 2:
    print('python script.py <group-file> <group-file> ...')
    return 1

  filenames = argv[1:]

  # read halos
  mass_list = []
  rad_list = []
  pos_list = []
  scales = []
  for file in filenames:
    mass,rad,pos,header = group_data(file,opts={'mass':True,'rad':True,'pos':True})
    mass_list += [mass]
    rad_list += [rad]
    pos_list += [pos]
    scales += [1./(1+header['Redshift'])]
    print('%d groups at a=%g'%(mass.size,scales[-1]))

  OmegaM = header['Omega0']
  OmegaL = header['OmegaLambda']
  HubbleParam = header['HubbleParam']

  scales = np.array(scales)
  times = np.zeros_like(scales)
  for i,a in enumerate(scales):
    times[i] = quad(dtda,scales[0],a,args=(HubbleParam,OmegaM))[0]

  nt = len(times)
  count = len(mass_list[-1])

  masses = np.zeros((nt,count))
  radii = np.zeros((nt,count))
  positions = np.zeros((nt,count,3))
  indices = np.zeros((nt,count,),dtype=int)-1

  masses[-1] = mass_list[-1]
  radii[-1] = rad_list[-1]
  positions[-1] = pos_list[-1]
  indices[-1] = np.arange(count)
  for j in range(nt-1,0,-1): # from nt-1 to 1
    print('matching time %g -> %g'%(times[j],times[j-1]))
    xnew = np.concatenate((pos_list[j-1],rad_list[j-1][:,None]),axis=1)
    x = np.concatenate((positions[j],radii[j,:,None]),axis=1)
    tree = KDTree(xnew)
    indices[j-1] = tree.query(x,k=1,p=2)[1]
    masses[j-1] = mass_list[j-1][indices[j-1]]
    radii[j-1] = rad_list[j-1][indices[j-1]]
    positions[j-1] = pos_list[j-1][indices[j-1]]

  np.savez('halos.npz',files=np.array(filenames),scale=scales,time=times,index=indices,mass=masses,radius=radii,position=positions)

if __name__ == '__main__':
  from sys import argv
  run(argv)                         
