import numpy as np
from scipy.fft import fftn,ifftn,fftshift
import h5py

def gadget_to_particles_DMO(filename):
  
  '''
  
  Read particles from GADGET HDF5 snapshot. Only reads particles of type 1 ("halo").
  
  Parameters:
    
    filename: input file name
    
  Returns:
    
    pos: position array, shape (3,NP), comoving
    
    vel: velocity array, shape (3,NP), peculiar
    
    mass: mass array, shape (NP)
    
    header: a dict with header info, use list(parameters) to see the fields
  
  '''
  
  with h5py.File(filename, 'r') as f:
    
    header = dict(f['Header'].attrs)
    
    MassTable = header['MassTable']
    ScaleFactor = 1./(1+header['Redshift'])
    NP = header['NumPart_ThisFile'][1]
    
    pos = np.array(f['PartType1/Coordinates']).reshape((NP,3)).T
    vel = np.array(f['PartType1/Velocities']).reshape((NP,3)).T * np.sqrt(ScaleFactor)
    if MassTable[1] == 0.:
      mass = np.array(f['PartType1/Masses'])
    else:
      mass = np.full(NP,MassTable[1])
    
    return pos, vel, mass, header

def fof_to_halos(filename):

  '''
  
  Read halos from GADGET HDF5 FOF file.
  
  Parameters:
    
    filename: input file name
    
  Returns:
    
    pos: position array, shape (3,NH), comoving
    
    vel: velocity array, shape (3,NH), peculiar
    
    mass: mass array, shape (NH)
    
  '''

  with h5py.File(filename, 'r') as f:

    header = dict(f['Header'].attrs)

    ScaleFactor = 1./(1+header['Redshift'])

    pos = np.array(f['Group/GroupPos']).T
    vel = np.array(f['Group/GroupVel']).T * np.sqrt(ScaleFactor)
    mass = np.array(f['Group/GroupMass'])

    return pos, vel, mass

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
  
  Bin particles into a density field using cloud-in-cell method
  
  Parameters:
    
    pos: 3D positions relative to center, shape (3,NP) where NP is the number of particles

    mass: masses of particles, shape (NP)
    
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
