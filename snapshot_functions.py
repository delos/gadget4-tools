import numpy as np
from scipy.fft import fftn,ifftn,fftshift
import h5py
from pathlib import Path

def gadget_to_particles(fileprefix):

  '''
  
  Read particles from GADGET HDF5 snapshot.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)
    
  Returns:
    
    pos: position array, shape (3,NP), comoving
    
    vel: velocity array, shape (3,NP), peculiar
    
    mass: mass array, shape (NP)
    
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
  pinst = 0
  while True:
    if fileinst >= numfiles:
      break

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      MassTable = header['MassTable']
      ScaleFactor = 1./(1+header['Redshift'])
      NP = header['NumPart_ThisFile']
      NPtot = header['NumPart_Total']
      numfiles = header['NumFilesPerSnapshot']

      if fileinst == 0:
        pos = np.zeros((3,np.sum(NPtot)))
        vel = np.zeros((3,np.sum(NPtot)))
        mass = np.zeros(np.sum(NPtot))

      for typ in range(len(NPtot)):
        NPtyp = int(NP[typ])
        if NPtyp == 0:
          continue

        pos[:,pinst:pinst+NPtyp] = np.array(f['PartType%d/Coordinates'%typ]).T
        vel[:,pinst:pinst+NPtyp] = np.array(f['PartType%d/Velocities'%typ]).T * np.sqrt(ScaleFactor)

        if MassTable[typ] == 0.:
          mass[pinst:pinst+NPtyp] = np.array(f['PartType%d/Masses'%typ])
        else:
          mass[pinst:pinst+NPtyp] = np.full(NPtyp,MassTable[typ])

        pinst += NPtyp

    fileinst += 1
    
  return pos, vel, mass, header

def fof_to_halos(fileprefix):

  '''
  
  Read halos from GADGET HDF5 FOF file.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_tab_000, not fof_tab_000.0.hdf5)
    
  Returns:
    
    pos: position array, shape (3,NH), comoving
    
    vel: velocity array, shape (3,NH), peculiar
    
    mass: mass array, shape (NH)

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
  pinst = 0
  pos = []
  vel = []
  mass = []
  while True:
    if fileinst >= numfiles:
      break

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      ScaleFactor = 1./(1+header['Redshift'])
      numfiles = header['NumFiles']

      pos += [np.array(f['Group/GroupPos']).T]
      vel += [np.array(f['Group/GroupVel']).T * np.sqrt(ScaleFactor)]
      mass += [np.array(f['Group/GroupMass'])]

    fileinst += 1

  return np.concatenate(pos,axis=1), np.concatenate(vel,axis=1), np.concatenate(mass), header

def cic_bin(x,BoxSize,GridSize,weights=1,density=True):
  
  '''
  
  Bin particles into a density field using cloud-in-cell method
  
  Parameters:
    
    x: 3D positions, shape (3,NP) where NP is the number of particles
    
    BoxSize: size of periodic region
    
    GridSize: resolution of output density field, per dimension
    
    weights: weight (e.g. mass) to assign to each particle, either a number or
    an array of length NP
    
    density: If False, output the total mass within each cell. If True, output
    mass/volume.
    
  Returns:
    
    field of shape (GridSize,GridSize,GridSize)
    
    bin edges
  
  '''
  
  NP = x.shape[1]
  
  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)
  
  # idea:
  # i and i1 are indices of the two adjacent cells (in each dimension)
  # f is the fraction from i to i1 where particle lies
  # shapes are (3,NP)
  
  f = x / dx
  
  f[f < 0.5] += N
  f[f >= N+0.5] -= N
  
  i = (f-0.5).astype(np.int32)
  f -= i + 0.5
  
  i1 = i+1
  
  i[i<0] += N
  i[i>=N] -= N
  i1[i1<0] += N
  i1[i1>=N] -= N
  
  # now appropriately add each particle into the 8 adjacent cells
  
  hist = np.zeros((N,N,N))
  np.add.at(hist,(i[0],i[1],i[2]),(1-f[0])*(1-f[1])*(1-f[2])*weights)
  np.add.at(hist,(i1[0],i[1],i[2]),f[0]*(1-f[1])*(1-f[2])*weights)
  np.add.at(hist,(i[0],i1[1],i[2]),(1-f[0])*f[1]*(1-f[2])*weights)
  np.add.at(hist,(i[0],i[1],i1[2]),(1-f[0])*(1-f[1])*f[2]*weights)
  np.add.at(hist,(i1[0],i1[1],i[2]),f[0]*f[1]*(1-f[2])*weights)
  np.add.at(hist,(i[0],i1[1],i1[2]),(1-f[0])*f[1]*f[2]*weights)
  np.add.at(hist,(i1[0],i[1],i1[2]),f[0]*(1-f[1])*f[2]*weights)
  np.add.at(hist,(i1[0],i1[1],i1[2]),f[0]*f[1]*f[2]*weights)
  
  if density:
    hist /= dx**3
    
  return hist,bins

def power_spectrum(delta,BoxSize,bins=None):
  
  '''
  
  Find spherically averaged power spectrum of density field
  
  Parameters:
    
    delta: input density field
    
    BoxSize: width of periodic box
    
    bins: desired k bin edges
    
  Returns:
    
    k: array of wavenumbers
    
    P(k): array comprising the power spectrum as a function of k
  
  '''
  
  GridSize = delta.shape[0]
  dk = 2*np.pi/BoxSize
  
  # radial bins for k
  if bins is None:
    # corner of cube is at distance np.sqrt(3)/2*length from center
    bins = np.arange(1,int((GridSize+1) * np.sqrt(3)/2)) * dk
  
  # get wavenumbers associated with k-space grid
  k = ((np.indices(delta.shape)+GridSize//2)%GridSize-GridSize//2) * dk
  k_mag = np.sqrt(np.sum(k**2,axis=0))
  
  # Fourier transform and get power spectrum
  pk = np.abs(fftn(delta,overwrite_x=True))**2*BoxSize**3/GridSize**6
  
  hist_pk,_ = np.histogram(k_mag,bins=bins,weights=pk)
  hist_ct,_ = np.histogram(k_mag,bins=bins)
  hist_k,_ = np.histogram(k_mag,bins=bins,weights=k_mag)
  
  return hist_k/hist_ct, hist_pk/hist_ct


def density_profile(pos,mass,bins=None,BoxSize=None):
  
  '''
  
  Spherically averaged density profile centered at position (0,0,0)
  
  Parameters:
    
    pos: 3D positions relative to center, shape (3,NP) where NP is the number of particles

    mass: masses of particles, shape (NP)

    bins: radial bin edges
    
    BoxSize: size of periodic region (None if not periodic)
    
  Returns:
    
    radius, density

  
  '''
  
  NP = pos.shape[1]

  # shift periodic box
  if BoxSize is not None:
    pos[pos >= 0.5*BoxSize] -= BoxSize
    pos[pos < -0.5*BoxSize] -= BoxSize

  # radii
  r = np.sqrt(np.sum(pos**2,axis=0))

  # radial bins
  if bins is None:
    rmin = np.sort(r)[100]/10
    rmax = np.max(r)
    bins = np.geomspace(rmin,rmax,50)
  bin_volume = 4./3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
  
  hist_mass,_ = np.histogram(r,bins=bins,weights=mass)

  return 0.5*(bins[1:]+bins[:-1]), hist_mass / bin_volume
