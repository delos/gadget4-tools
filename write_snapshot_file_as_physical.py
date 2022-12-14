import numpy as np
import h5py
from scipy.integrate import quad

def get_H(a,H0,OmegaM,OmegaL,OmegaR):
  return H0 * np.sqrt(OmegaM*a**-3 + OmegaL + OmegaR*a**-4) # unitV / unitL

def dtda_fun(a,H0,OmegaM,OmegaL,OmegaR):
  return 1./(a*get_H(a,H0,OmegaM,OmegaL,OmegaR)) # unitL / unitV

def time(a,H0,OmegaM,OmegaL,OmegaR):
  return quad(dtda_fun,0,a,args=(H0,OmegaM,OmegaL,OmegaR,))[0] # unitL / unitV

def run(argv):
  
  if len(argv) < 3:
    print('python script.py <snapshot-file> <out>')
    return 1

  headerNew = {}
  with h5py.File(argv[1], 'r') as f:
    header = dict(f['Header'].attrs)
    param  = dict(f['Parameters'].attrs)
    cosmo = [
      param['Hubble'],
      param['Omega0'] + (param.get('OmegaSmooth') or 0.),
      param['OmegaLambda'],
      param.get('OmegaR') or 0.,
      ]
    a = 1./(1+header['Redshift'])
    H = get_H(a,*cosmo)
    t = time(a,*cosmo)
    print('a = %g -> time = %g'%(a,t))
    with h5py.File(argv[2], 'w') as fNew:
      headerNew = fNew.create_group('Header')
      headerNew.attrs['MassTable'] = header['MassTable']
      headerNew.attrs['NumPart_ThisFile'] = header['NumPart_ThisFile']
      headerNew.attrs['NumPart_Total'] = header['NumPart_Total']
      headerNew.attrs['NumFilesPerSnapshot'] = header['NumFilesPerSnapshot']
        
      for typ in range(header['MassTable'].size):
        if header['NumPart_ThisFile'][typ] == 0:
          continue
        particles = fNew.create_group('PartType%d'%typ)
        particles.create_dataset('Coordinates',
          data=(f['PartType%d/Coordinates'%typ][:]*a).astype(np.float32))
        particles.create_dataset('Velocities',
          data=(f['PartType%d/Velocities'%typ][:]*np.sqrt(a) + f['PartType%d/Coordinates'%typ][:]*a*H).astype(np.float32))
        particles.create_dataset('Masses',
          data=f['PartType%d/Masses'%typ])
        particles.create_dataset('ParticleIDs',
          data=f['PartType%d/ParticleIDs'%typ])

  return 0

if __name__ == '__main__':
  from sys import argv
  run(argv)                         

