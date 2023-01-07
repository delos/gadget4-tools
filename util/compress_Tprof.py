import numpy as np

i = 0

d = []
e = []
p = []

while True:
  if (i+1)%1000 == 0:
    print(i+1)
  try:
    r, d_, e_, p_ = np.loadtxt('tidal_tensor_profile_%04d.txt'%i,dtype=np.float32).T
  except:
    print('%d peaks'%i)
    break
  d += [d_]
  e += [e_]
  p += [p_]
  i += 1

np.savez('tidal_tensor_profiles.npz',r=r,d=np.array(d),e=np.array(e),p=np.array(p))
