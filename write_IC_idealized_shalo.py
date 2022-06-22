import numpy as np
import h5py
from scipy.integrate import quad
from numba import njit, vectorize
import matplotlib.pyplot as plt

# mass function helpers

def mass_coef(n,mmin,mmax,a):
  return (a-1.)*n/(mmin**(1.-a)-mmax**(1.-a))

def sample_mass(N, mmin, mmax, a):
  p = np.random.rand(N)
  return mmin*(1 - (1 - (mmax/mmin)**(1 - a))*p)**(1/(1 - a))

def mean_mass(mmin, mmax, a):
  if a == 2:
    return np.log(mmax/mmin)/(1./mmin-1./mmax)
  return (a-1.)/(2.-a) * (mmax**(2.-a)-mmin**(2.-a))/(mmin**(1.-a)-mmax**(1.-a))

# power spectrum helpers

@vectorize
def rhok(x):
  y = 0.7 * x
  if y < 1e-2:
    return 1. + y**2*(-0.6 + 0.1619047619047619*y**2)
  return 3*np.sin(y)**3*(np.sin(y)-y*np.cos(y))/y**6

@njit
def dPdlnM(lnM,k,Mcoef,Rcoef,alpha):
  M = np.exp(lnM)
  R = Rcoef*M**(1./3)
  return Mcoef * M**(3-alpha) * rhok(k*R)**2

def compute_pk(boxsize,N,Mcoef,Rcoef,alpha,Mmin,Mmax):
  kmin = 2*np.pi/boxsize
  kmax = np.sqrt(3)*N**(1./3)*np.pi/boxsize
  ks = np.geomspace(kmin,kmax,200)
  Ps = np.zeros_like(ks)
  for i,k in enumerate(ks):
    Ps[i] = quad(dPdlnM,np.log(Mmin),np.log(Mmax),args=(k,Mcoef,Rcoef,alpha),limit=5000,epsabs=1e-60)[0]
  np.savetxt('power.txt',
    np.stack((ks,ks**3/(2*np.pi**2)*Ps)).T,
    header='k (kpc^-1),  rho^2 P (Msun^2 kpc^-6)')
  print('saved power to power.txt')
  return ks,Ps

# main

def run(argv):
  G = 4.30241002e-6 # (km/s)^2 kpc/Msun
  
  try:
    nstar = float(argv[1])
    n, mmin, mmax, alpha, rho = [float(x) for x in argv[2].split(',')]
    sigma = float(argv[3])
    boxsize = float(argv[4])
  except Exception as e:
    print(e)
    print()
    print('python script.py <star n> <subhalo n,Mmin,Mmax,alpha,rho> <sigma> <box size>')
    return

  # number of particles
  N = np.random.poisson(n * boxsize**3)
  Nstar = np.random.poisson(nstar * boxsize**3)

  print('%d stars (%g expected)'%(Nstar, nstar * boxsize**3))
  print('%d halos (%g expected)'%(N, n * boxsize**3))

  # masses
  mstar = 1. # shouldn't matter. n=rho for convenience.
  m = sample_mass(N, mmin, mmax, alpha)

  Mcoef = mass_coef(n,mmin,mmax,alpha)
  Mmean = np.mean(m)
  print('dn/dM = %g M^-%f  from  %e to %e'%(Mcoef,alpha,mmin,mmax))
  print('mean halo mass = %e (%e)'%(Mmean,mean_mass(mmin,mmax,alpha)))

  rho_mean = Mmean * N / boxsize**3

  # radii
  M_over_R3 = rho * 343.*np.pi/125.
  Rcoef = M_over_R3**(-1./3)
  print('R = %g M^{1/3}'%Rcoef)

  # power
  k,P = compute_pk(boxsize,Nstar,Mcoef,Rcoef,alpha,mmin,mmax)
  
  # gaussian velocities
  s = sigma / np.sqrt(3.) # 3D -> 1D
  vstar = np.random.normal(0,s,(Nstar,3))
  v = np.random.normal(0,s,(N,3))

  print('3D velocity dispersion = %g => 1D velocity dispersion = %g'%(sigma,s))
  print('  stars: %g'%(np.sqrt(np.mean(vstar**2))))
  print('  halos: %g'%(np.sqrt(np.mean(v**2))))

  print('  k_J L = %f'%(np.sqrt(4*np.pi*G*rho_mean)/sigma*boxsize))

  # uniformly distribute positions
  xstar = np.random.uniform(0,boxsize,(Nstar,3))
  x = np.random.uniform(0,boxsize,(N,3))
  
  # write to snapshot
  with h5py.File('init.hdf5', 'w') as f:

    header = f.create_group('Header')
    header.attrs['NumPart_ThisFile'] = np.array([0,Nstar,N],dtype=np.uint32)
    header.attrs['NumPart_Total'] = np.array([0,Nstar,N],dtype=np.uint64)
    header.attrs['MassTable'] = np.array([0,mstar,0],dtype=np.float64) # mass=0 => look for mass block
    header.attrs['Time'] = np.array(0,dtype=np.float64) # doesn't matter
    header.attrs['Redshift'] = np.array(0,dtype=np.float64) # doesn't matter
    header.attrs['BoxSize'] = np.array(boxsize,dtype=np.float64) # doesn't matter
    header.attrs['NumFilesPerSnapshot'] = np.array(1,dtype=np.int32)

    stars = f.create_group('PartType1') # type 0 is reserved for gas
    starx = stars.create_dataset('Coordinates', data=xstar.astype(np.float32))
    starv = stars.create_dataset('Velocities', data=vstar.astype(np.float32))
    stari = stars.create_dataset('ParticleIDs', data=np.arange(Nstar,dtype=np.uint32))
    halos = f.create_group('PartType2') # type 0 is reserved for gas
    halox = halos.create_dataset('Coordinates', data=x.astype(np.float32))
    halov = halos.create_dataset('Velocities', data=v.astype(np.float32))
    haloi = halos.create_dataset('ParticleIDs', data=np.arange(Nstar,Nstar+N,dtype=np.uint32))
    halom = halos.create_dataset('Masses', data=m.astype(np.float32))

  #
  t = boxsize / s
  NR = 125
  Rmin = Rcoef*mmin**(1./3)
  Rmax = Rcoef*mmax**(1./3)
  Rs = np.geomspace(Rmin,Rmax,NR)
  print('radii spaced by factor %f from %e to %e'%(Rs[1]/Rs[0],Rmin,Rmax))

  # parameters
  with open('param.txt','wt') as f:
    f.write('OutputDir                ./output\n')
    f.write('SnapshotFileBase         snapshot\n')
    f.write('SnapFormat               3\n')
    f.write('ICFormat                 3\n')
    f.write('InitCondFile             ./init\n')
    f.write('NumFilesPerSnapshot      1\n')
    f.write('MaxFilesWithConcurrentIO 8\n')
    f.write('TimeLimitCPU             86100.0\n')
    f.write('CpuTimeBetRestartFile    7200\n')
    f.write('MaxMemSize               2000\n')
    f.write('TimeBegin                0.0\n')
    f.write('TimeMax                  %.5e\n'%(t*10))
    f.write('ComovingIntegrationOn    0\n')
    f.write('BoxSize                  %lg\n'%boxsize)
    f.write('Omega0                   1.0\n')
    f.write('OmegaLambda              0.0\n')
    f.write('OmegaBaryon              0.0\n')
    f.write('HubbleParam              1.0\n')
    f.write('Hubble                   0.1\n')
    f.write('UnitLength_in_cm         3.085678e21\n')
    f.write('UnitMass_in_g            1.989e33\n')
    f.write('UnitVelocity_in_cm_per_s 1.0e5\n')
    f.write('GravityConstantInternal  0\n')
    f.write('TypeOfOpeningCriterion   1\n')
    f.write('ErrTolTheta              0.7\n')
    f.write('ErrTolForceAcc           0.005\n')
    f.write('ErrTolThetaMax           1.0\n')
    f.write('MaxSizeTimestep          %.5e\n'%(t*.1))
    f.write('MinSizeTimestep          0\n')
    f.write('ErrTolIntAccuracy        0.025\n')
    f.write('CourantFac               0.15\n')
    f.write('ActivePartFracForNewDomainDecomp 0.01\n')
    f.write('TopNodeFactor            2.5\n')
    f.write('OutputListOn             0\n')
    f.write('OutputListFilename       outputs.txt\n')
    f.write('TimeOfFirstSnapshot      0.0\n')
    f.write('TimeBetSnapshot          %.5e\n'%(t*.1))
    f.write('TimeBetStatistics        %.5e\n'%t)
    f.write('DesNumNgb                64\n')
    f.write('MaxNumNgbDeviation       2\n')
    f.write('InitGasTemp              10000\n')
    f.write('ArtBulkViscConst         1.0\n')
    f.write('MinEgySpec               0\n')
    f.write('SofteningClassOfPartType0     0\n')
    f.write('SofteningClassOfPartType1     0\n')
    f.write('SofteningClassOfPartType2     1\n')
    f.write('SofteningComovingClass0       %lg\n'%(boxsize*1e-5))
    f.write('SofteningMaxPhysClass0        %lg\n'%(boxsize*1e-5))
    f.write('SofteningComovingClass1       %lg\n'%(Rcoef*Mmean**(1./3)))
    f.write('SofteningMaxPhysClass1        %lg\n'%(Rcoef*Mmean**(1./3)))
    for i,R in enumerate(Rs):
      f.write('SofteningComovingClass%d       %lg\n'%(i+2,R))
      f.write('SofteningMaxPhysClass%d        %lg\n'%(i+2,R))

  # config
  with open('Config.sh','wt') as f:
    f.write('PERIODIC\n')
    f.write('NTYPES=3\n')
    f.write('SELFGRAVITY\n')
    f.write('INDIVIDUAL_GRAVITY_SOFTENING=4\n')
    f.write('NSOFTCLASSES=%d\n'%(NR+2))
    f.write('DOUBLEPRECISION=1\n')

  np.savez('params.npz',box=boxsize,s=s,n=nstar)
  plt.loglog(k,4*2**.5*np.pi**3*G**2/(s**4*k**4)*k**3/(2*np.pi**2)*P)
  plt.loglog(k,k**3/(2*np.pi**2)/nstar,color='k',ls='--')
  plt.axhline(1,ls=':',color='k')
  plt.xlabel(r'$k$')
  plt.ylabel(r'$\mathcal{P}(k)$')
  plt.show()
  return

  
if __name__ == '__main__':
  from sys import argv
  run(argv)
