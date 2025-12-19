import numpy as np
from numba import njit
from snapshot_functions import read_particles_filter, read_subhalos

@njit
def profile(pos,mass,rmin,rmax,nr):

  bin_ct = np.zeros(nr,dtype=np.int32)
  bin_m = np.zeros(nr)
  low_ct = 0
  low_m = 0.

  logrmaxrmin = np.log(rmax/rmin)

  for p in range(mass.size):
    r = np.sqrt(np.sum(pos[p]**2))

    if r >= rmax:
      continue
    if r < rmin:
      low_ct += 1
      low_m += mass[p]
      continue

    i = int(np.log(r/rmin)/logrmaxrmin * nr)
    if i < 0 or i >= nr:
      raise Exception('...')

    bin_ct[i] += 1
    bin_m[i] += mass[p]

  bin_rl = rmin * (rmax/rmin)**(np.arange(nr)*1./nr)
  bin_ru = rmin * (rmax/rmin)**(np.arange(1,nr+1)*1./nr)
  bin_r = (0.5*(bin_rl**3 + bin_ru**3))**(1./3) # volume-averaged radius

  bin_vol = 4./3 * np.pi * (bin_ru**3-bin_rl**3)

  bin_rho = bin_m / bin_vol

  bin_mcum = np.cumsum(bin_m) + low_m

  bin_ctcum = np.cumsum(bin_ct) + low_ct

  return bin_r, bin_rho, bin_ru, bin_mcum, bin_ctcum

def run(argv):
  
  if len(argv) < 4:
    print('python script.py <snapshot-file> <group-file> <rmin,rmax,nr> [types=-1] [outname] [top N]')
    return 1
  
  rb = argv[3].split(',')
  rmin = float(rb[0])
  rmax = float(rb[1])
  nr = int(rb[2])
  
  try:
    types = [int(x) for x in argv[4].split(',')]
    if np.any(np.array(types)<0):
      types = None
    print('particle types ' + ', '.join([str(x) for x in types]))
  except:
    types = None

  try:
    outname = argv[5]
    if outname[-4:] != '.npz':
      outname += '.npz'
  except:
    outname = argv[1] + '_grpprof.npz'

  try: count = int(argv[6])
  except: count = None

  # read halos
  hpos, hmass, hrad, header = read_subhalos(argv[2],
    opts={},group_opts={'pos':True,'somass':True,'soradius':True,},
    )
  NH = hmass.shape[0]

  if count is not None and count < NH:
    idxh = np.argsort(hmass)[::-1][:count]
  else:
    idxh = np.argsort(hmass)[::-1]
  nwrite = idxh.size
  print('%d halos out of %d'%(nwrite,NH))

  rlist = None
  hmasslist = np.zeros(nwrite,dtype=np.float32)
  hradlist = np.zeros(nwrite,dtype=np.float32)
  hposlist = np.zeros((nwrite,3),dtype=np.float32)
  hvellist = np.zeros((nwrite,3),dtype=np.float32)

  for i in range(NH): # loop over groups
    if i in idxh:
      print('group %d [mass %.3g]'%(i,hmass[i],))
      # read particles
      try:
        pos, mass, header = read_particles_filter(argv[1],center=hpos[i],radius=hrad[i],type_list=types,opts={'mass':True,'pos':True})
      except:
        print('no particles; skip')
        continue

      # get profile
      r, rho, ru, m, ct = profile(pos,mass,rmin,rmax,nr)

      if rlist is None:
        rlist = r
        rholist = np.zeros((nwrite,nr),dtype=np.float32)
        rulist = ru
        mlist = np.zeros((nwrite,nr),dtype=np.float32)
        ctlist = np.zeros((nwrite,nr),dtype=np.int32)

      idx = np.where(idxh == i)[0][0]

      rholist[idx] = rho
      mlist[idx] = m
      ctlist[idx] = ct
      
      hmasslist[idx] = hmass[i]
      hradlist[idx] = hrad[i]
      hposlist[idx] = hpos[i]

  np.savez(outname,r=rlist,rho=rholist,ru=rulist,m=mlist,N=ctlist,M=hmasslist,R=hradlist,X=hposlist,a=1./(1+header['Redshift']))

if __name__ == '__main__':
  from sys import argv
  run(argv)
