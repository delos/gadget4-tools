import numpy as np
from scipy.fft import fftn,ifftn,fftshift
import h5py

def make_field(pkfun,BoxSize,GridSize):
  
  '''
  
  Generate random periodic Gaussian field with given power spectrum.
  
  Use even GridSize. I'd have to think about whether this works for odd GridSize.
  
  Parameters:
    
    pkfun: function that gives the power spectrum as a function of wavenumber
    
    BoxSize: physical size of the periodic box
    
    GridSize: number of elements of output grid, per dimension
    
  Returns:
    
    discrete field, shape (GridSize,GridSize,GridSize)
  
  '''
  
  # first make unit random noise
  delta = np.random.normal(size=(GridSize,GridSize,GridSize))
  delta -= delta.mean()
  delta /= np.sqrt(np.mean(delta**2))
  deltak = fftn(delta,overwrite_x=True)/GridSize**1.5
  
  # get wavenumbers associated with k-space grid
  dk = 2*np.pi/BoxSize
  k = ((np.indices(deltak.shape)+GridSize//2)%GridSize-GridSize//2) * dk
  k_mag = np.sqrt(np.sum(k**2,axis=0))
                  
  # scale random noise by power spectrum
  deltak *= np.sqrt(pkfun(k_mag)*GridSize**6/BoxSize**3)
  
  '''
  
  Note: an alternative procedure is to do the random sampling directly in k-space.
  
  '''
  
  return np.real(ifftn(deltak,overwrite_x=True))

def field_to_particles(delta,BoxSize,HubbleRate):
  
  '''
  
  Generate particle positions and velocities from a periodic density grid.
  
  For the velocities, we assume matter domination.
  
  Parameters:
    
    delta: fractional density contrast grid, shape (GridSize,GridSize,GridSize)
    
    BoxSize: length of periodic box
    
    HubbleRate: H(a) at the scale factor a at which particles are generated
    
  Returns:
    
    pos, vel: arrays of shape (3,GridSize**3) representing the 3D positions and
    velocities ofGridSize**3 particles. Velocities are peculiar velocities.
  
  '''
  
  GridSize = delta.shape[0]
  dk = 2*np.pi/BoxSize
  dx = BoxSize/GridSize
  
  # first get grid in k-space
  deltak = fftn(delta,overwrite_x=True)
  k = ((np.indices(deltak.shape)+GridSize//2)%GridSize-GridSize//2) * dk
  k2 = np.sum(k**2,axis=0)
  
  # get displacement from delta
  # displacement = i (k/k^2) delta if the displacement is irrotational
  dispk = 1j * k * deltak[None,:,:,:] / k2[None,:,:,:]
  dispk[:,0,0,0] = 0.
  disp = np.real(ifftn(dispk,axes=(1,2,3),overwrite_x=True))
  
  # make positions and velocities
  pos = disp + (np.indices(deltak.shape)+0.5)*dx
  vel = disp * HubbleRate
  
  # keep particles inside box, shouldn't be an issue if delta<1 though
  pos[pos < 0] += BoxSize
  pos[pos >= BoxSize] -= BoxSize
  
  return pos.reshape(3,-1), vel.reshape(3,-1)

def particles_to_gadget_comoving_periodic(filename,pos,vel,BoxSize,ScaleFactor,H0=0.06774,OmegaM=0.3089,G=4.3022682e-6):
  
  '''
  
  Generate GADGET HDF5 snapshot from position and velocity arrays.
  
  We assume a periodic comoving simulation and set particle masses accordingly.
  
  Make sure units are consistent between pos, vel, H0, and G. Default H0 and G values
  imply mass unit of Msun, length unit of kpc, and velocity unit of km/s.
  
  Parameters:
    
    filename: output file name
    
    pos: 3D positions, shape (3,NP) where NP is the number of particles
    
    vel: 3D velocities, shape (3,NP)
    
    BoxSize: physical size of the periodic box
    
    ScaleFactor: cosmological scale factor at which to make snapshot
    
    H0: Hubble rate today (default value is in km/s/kpc)
    
    OmegaM: matter density today in units of the critical density
    
    G: Newton's constant (default value is in (km/s)^2 kpc/Msun)
  
  '''
  
  NP = pos.shape[1]
  
  # comoving matter density
  rho = 3*H0**2/(8*np.pi*G) * OmegaM
  
  # mass per particle
  ParticleMass = rho * BoxSize**3 / NP
  
  # save to HDF5 file
  with h5py.File(filename, 'w') as f:
    
    header = f.create_group('Header')
    header.attrs['NumPart_ThisFile'] = np.array([0,NP,0,0,0,0],dtype=np.uint32)
    header.attrs['NumPart_Total'] = np.array([0,NP,0,0,0,0],dtype=np.uint64)
    header.attrs['MassTable'] = np.array([0,ParticleMass,0,0,0,0],dtype=np.float64)
    header.attrs['Time'] = np.array(ScaleFactor,dtype=np.float64) # GADGET doesn't read this
    header.attrs['Redshift'] = np.array(1./ScaleFactor-1,dtype=np.float64) # or this
    header.attrs['BoxSize'] = np.array(BoxSize,dtype=np.float64) # or this
    header.attrs['NumFilesPerSnapshot'] = np.array(1,dtype=np.int32)
    
    particles = f.create_group('ParticleType1') # type 0 is reserved for gas
    coodinates = particles.create_dataset('Coordinates', data=(pos.T.flatten()).astype(np.float32))
    velocities = particles.create_dataset('Velocities', # sqrt(a) is GADGET convention
                                          data=((vel/np.sqrt(ScaleFactor)).T.flatten()).astype(np.float32))
    ids = particles.create_dataset('ParticleIDs', data=np.arange(NP,dtype=np.uint32))

def power_to_gadget(filename,pkfun,BoxSize,GridSize,ScaleFactor,H0=0.06774,OmegaM=0.3089,G=4.3022682e-6):
  
  '''
  
  Generate GADGET HDF5 snapshot from power spectrum
  
  Make sure units are consistent between pkfun, H0, and G. Default H0 and G values
  imply mass unit of Msun, length unit of kpc, and velocity unit of km/s.
  
  Parameters:
    
    filename: output file name
    
    pkfun: function that gives the power spectrum as a function of wavenumber
    
    BoxSize: physical size of the periodic box
    
    GridSize: number of elements of output grid, per dimension
    (may have to be even)
    
    ScaleFactor: cosmological scale factor at which to make snapshot
    
    H0: Hubble rate today (default value is in km/s/kpc)
    
    OmegaM: matter density today in units of the critical density
    
    G: Newton's constant (default value is in (km/s)^2 kpc/Msun)
  
  '''
  
  # Hubble rate at specified scale factor
  HubbleRate = H0 * np.sqrt(OmegaM * ScaleFactor**-3)
  
  # get density field
  delta = make_field(pkfun,BoxSize,GridSize)
  
  # get particles
  # we could optimize by not transforming delta into position space at all
  pos,vel = field_to_particles(delta,BoxSize,GridSize,HubbleRate)
  
  particles_to_gadget_comoving_periodic(filename,pos,vel,BoxSize,ScaleFactor,H0,OmegaM,G)

def particles_to_gadget_massrange(filename,pos,vel,mass):
  
  '''
  
  Generate GADGET HDF5 snapshot from position, velocity, and mass arrays.
  
  Parameters:
    
    filename: output file name
    
    pos: 3D positions, shape (3,NP) where NP is the number of particles
    
    vel: 3D velocities, shape (3,NP)
    If using comoving integration, these should be (peculiar velocity)/sqrt(a)
    
    mass: particle masses, shape (NP)
    If using comoving periodic integration, masses must add up to average mass
    within box.
  
  '''
  
  NP = pos.shape[1]
  
  # save to HDF5 file
  with h5py.File(filename, 'w') as f:
    
    header = f.create_group('Header')
    header.attrs['NumPart_ThisFile'] = np.array([0,NP,0,0,0,0],dtype=np.uint32)
    header.attrs['NumPart_Total'] = np.array([0,NP,0,0,0,0],dtype=np.uint64)
    header.attrs['MassTable'] = np.array([0,0,0,0,0,0],dtype=np.float64) # mass=0 => look for mass block
    header.attrs['Time'] = np.array(0,dtype=np.float64) # doesn't matter
    header.attrs['Redshift'] = np.array(0,dtype=np.float64) # doesn't matter
    header.attrs['BoxSize'] = np.array(0,dtype=np.float64) # doesn't matter
    header.attrs['NumFilesPerSnapshot'] = np.array(1,dtype=np.int32)
    
    particles = f.create_group('ParticleType1') # type 0 is reserved for gas
    coodinates = particles.create_dataset('Coordinates', data=(pos.T.flatten()).astype(np.float32))
    velocities = particles.create_dataset('Velocities', data=(vel.T.flatten()).astype(np.float32))
    ids = particles.create_dataset('ParticleIDs', data=np.arange(NP,dtype=np.uint32))
    mass = particles.create_dataset('Masses', data=mass.astype(np.float32))