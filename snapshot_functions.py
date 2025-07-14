import numpy as np
from scipy.fft import fftn,ifftn,fftshift
import h5py
from pathlib import Path

fileprefix_snapshot = 'snapdir_%03d/snapshot_%03d'
fileprefix_subhalo = 'groups_%03d/fof_subhalo_tab_%03d'
fileprefix_subhalo_desc = 'groups_%03d/subhalo_desc_%03d'
fileprefix_subhalo_prog = 'groups_%03d/subhalo_prog_%03d'
file_extension = '.hdf5'

def _get_filebase(fileprefix):
  filepath = [
    Path(fileprefix + file_extension),
    Path(fileprefix + '.0' + file_extension),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + file_extension
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d' + file_extension
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1
  else:
    print(filepath[0],filepath[1],filepath[2])
    raise FileNotFoundError(fileprefix)
  return filebase, numfiles

def gadget_to_particles(fileprefix, opts={'pos':True,'vel':True,'ID':False,'mass':True}):

  '''
  
  Read particles from GADGET HDF5 snapshot.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)

    opts: which fields to read and return
    
  Returns:
    
    pos: position array, shape (3,NP), comoving
    
    vel: velocity array, shape (3,NP), peculiar

    ID: ID array, shape (NP,)
    
    mass: mass array, shape (NP,)
    
    header: a dict with header info, use list(header) to see the fields
  
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  fileinst = 0
  pinst = 0
  while fileinst < numfiles:

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
        # allocate full-sized memory blocks in advance, for efficiency
        if opts.get('pos'): pos = np.zeros((3,np.sum(NPtot)),dtype=np.float32)
        if opts.get('vel'): vel = np.zeros((3,np.sum(NPtot)),dtype=np.float32)
        if opts.get('mass'): mass = np.zeros(np.sum(NPtot),dtype=np.float32)
        if opts.get('ID'): ID = np.zeros(np.sum(NPtot),dtype=np.uint32)
 
      for typ in range(len(NPtot)):
        NPtyp = int(NP[typ])
        if NPtyp == 0:
          continue

        if opts.get('pos'): pos[:,pinst:pinst+NPtyp] = np.array(f['PartType%d/Coordinates'%typ]).T
        if opts.get('vel'): vel[:,pinst:pinst+NPtyp] = np.array(f['PartType%d/Velocities'%typ]).T * np.sqrt(ScaleFactor)

        if opts.get('mass'):
          if MassTable[typ] == 0.:
            mass[pinst:pinst+NPtyp] = np.array(f['PartType%d/Masses'%typ])
          else:
            mass[pinst:pinst+NPtyp] = np.full(NPtyp,MassTable[typ])

        if opts.get('ID'): ID[pinst:pinst+NPtyp] = np.array(f['PartType%d/ParticleIDs'%typ])

        pinst += NPtyp

    fileinst += 1

  ret = []
  if opts.get('pos'): ret += [pos]
  if opts.get('vel'): ret += [vel]
  if opts.get('mass'): ret += [mass]
  if opts.get('ID'): ret += [ID]
  ret += [header]

  return tuple(ret)

def fof_to_halos(fileprefix,opts={'pos':True,'vel':True,'mass':True}):

  '''
  
  Read halos from GADGET HDF5 FOF file.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_tab_000, not fof_tab_000.0.hdf5)

    opts: which fields to read and return
    
  Returns:
    
    pos: position array, shape (NH,3), comoving
    
    vel: velocity array, shape (NH,3), peculiar
    
    mass: mass array, shape (NH,)

    header: a dict with header info, use list(header) to see the fields
    
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  fileinst = 0
  if opts.get('pos'): pos = []
  if opts.get('vel'): vel = []
  if opts.get('mass'): mass = []
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

      if header['Ngroups_Total'] == 0:
        if opts.get('pos'): pos = [[]]
        if opts.get('vel'): vel = [[]]
        if opts.get('mass'): mass = [[]]
        break
 
      if header['Ngroups_ThisFile'] > 0:
        if opts.get('pos'): pos += [np.array(f['Group/GroupPos'])]
        if opts.get('vel'): vel += [np.array(f['Group/GroupVel']) * np.sqrt(ScaleFactor)]
        if opts.get('mass'): mass += [np.array(f['Group/GroupMass'])]

    fileinst += 1

  ret = []
  if opts.get('pos'): ret += [np.concatenate(pos,axis=1)]
  if opts.get('vel'): ret += [np.concatenate(vel,axis=1)]
  if opts.get('mass'): ret += [np.concatenate(mass)]
  ret += [header]

  return tuple(ret)

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

def subhalo_tracing_data(snapshot_number,subhalo_number):

  '''
  
  Get a subhalo's mass, position, velocity, progenitor, descendant,
  and other tracking information.
  
  Parameters:
    
    snapshot_number

    subhalo_number
    
  Returns:
    
    prog: best-scoring progenitor

    desc: best-scoring descendant

    pos: subhalo (comoving) position, shape (3,)

    vel: subhalo (peculiar) velocity, shape (3,)

    mass: subhalo mass
 
    ID: dict with group ID, subhalo ID, and most bound particle ID
  
    header: a dict with header info, use list(header) to see the fields
    
  '''

  prefix_sub = fileprefix_subhalo%tuple([snapshot_number]*fileprefix_subhalo.count('%'))
  prefix_desc = fileprefix_subhalo_desc%tuple([snapshot_number]*fileprefix_subhalo_desc.count('%'))
  prefix_prog = fileprefix_subhalo_prog%tuple([snapshot_number]*fileprefix_subhalo_prog.count('%'))

  filepath = [
    Path(prefix_sub + file_extension),
    Path(prefix_sub + '.0' + file_extension),
    Path(prefix_sub.split('/')[-1] + file_extension),
    ]

  if filepath[0].is_file():
    filebase_sub = prefix_sub + file_extension
    filebase_desc = prefix_desc + file_extension
    filebase_prog = prefix_prog + file_extension
    numfiles = 1
  elif filepath[1].is_file():
    filebase_sub = prefix_sub + '.%d' + file_extension
    filebase_desc = prefix_desc + '.%d' + file_extension
    filebase_prog = prefix_prog + '.%d' + file_extension
    numfiles = 2
  elif filepath[2].is_file():
    filebase_sub = prefix_sub.split('/')[-1] + file_extension
    filebase_desc = prefix_desc.split('/')[-1] + file_extension
    filebase_prog = prefix_prog.split('/')[-1] + file_extension
    numfiles = 1
  
  prog, desc, pos, vel, mass, ID, header = -1, -1, np.zeros(3), np.zeros(3), 0., {}, {}

  fileinst = 0
  hinst = 0
  ginst = 0
  while fileinst < numfiles:

    if numfiles == 1:
      filename_sub = filebase_sub
      filename_desc = filebase_desc
      filename_prog = filebase_prog
    else:
      filename_sub = filebase_sub%fileinst
      filename_desc = filebase_desc%fileinst
      filename_prog = filebase_prog%fileinst

    with h5py.File(filename_sub, 'r') as f:
      print('reading %s'%filename_sub)

      header = dict(f['Header'].attrs)

      ScaleFactor = 1./(1+header['Redshift'])
      numfiles = header['NumFiles']

      # is subhalo in this file?
      if hinst + header['Nsubhalos_ThisFile'] > subhalo_number:
        index = subhalo_number - hinst

        # get some data for the subhalo
        pos = np.array(f['Subhalo/SubhaloPos'])[index]
        vel = np.array(f['Subhalo/SubhaloVel'])[index] * np.sqrt(ScaleFactor)
        ID = {'group':np.array(f['Subhalo/SubhaloGroupNr'])[index],
          'subhalo':subhalo_number,
          'particle':np.array(f['Subhalo/SubhaloIDMostbound'])[index]}

        # get the subhalo and group mass. Group mass might be in an earlier file...
        if ID['group'] < ginst:
          ginst2 = ginst
          fileinst2 = fileinst
          while ID['group'] < ginst2:
            fileinst2 -= 1
            filename_sub2 = filebase_sub%fileinst2
            with h5py.File(filename_sub2, 'r') as f2:
              print('reading %s [back]'%filename_sub2)
              header2 = dict(f2['Header'].attrs)
              ginst2 -= int(header2['Ngroups_ThisFile'])
              if ID['group'] >= ginst2:
                mass = {'group':np.array(f2['Group/GroupMass'])[ID['group']-ginst2],
                  'subhalo':np.array(f['Subhalo/SubhaloMass'])[index]}
        elif ID['group'] >= ginst + int(header['Ngroups_ThisFile']): # or later file.
          ginst2 = ginst
          fileinst2 = fileinst
          ng2 = int(header['Ngroups_ThisFile'])
          while ID['group'] >= ginst2 + ng2:
            fileinst2 += 1
            filename_sub2 = filebase_sub%fileinst2
            with h5py.File(filename_sub2, 'r') as f2:
              print('reading %s [ahead]'%filename_sub2)
              header2 = dict(f2['Header'].attrs)
              ginst2 += ng2 
              ng2 = int(header2['Ngroups_ThisFile'])
              if ID['group'] < ginst2 + ng2:
                mass = {'group':np.array(f2['Group/GroupMass'])[ID['group']-ginst2],
                  'subhalo':np.array(f['Subhalo/SubhaloMass'])[index]}
        else:
          mass = {'group':np.array(f['Group/GroupMass'])[ID['group']-ginst],
            'subhalo':np.array(f['Subhalo/SubhaloMass'])[index]}

        # get best-matching descendent (decided already by SUBFIND)
        try:
          with h5py.File(filename_desc, 'r') as fd:
            print('reading %s'%filename_desc)
            if np.array(fd['Subhalo/SubhaloNr'])[index] != subhalo_number:
              raise Exception('halo number mismatch, %d != %d'%(np.array(fd['Subhalo/SubhaloNr'])[index],subhalo_number))
            desc = np.array(fd['Subhalo/DescSubhaloNr'])[index]
        except Exception as e:
          print(str(e))
          desc = -1

        # get best-matching progenitor (decided already by SUBFIND)
        try:
          with h5py.File(filename_prog, 'r') as fp:
            print('reading %s'%filename_prog)
            if np.array(fp['Subhalo/SubhaloNr'])[index] != subhalo_number:
              raise Exception('halo number mismatch, %d != %d'%(np.array(fp['Subhalo/SubhaloNr'])[index],subhalo_number))
            prog = np.array(fp['Subhalo/ProgSubhaloNr'])[index]
        except Exception as e:
          print(str(e))
          prog = -1

        break

      # advance halo and group indices as we advance in file number
      hinst += int(header['Nsubhalos_ThisFile'])
      ginst += int(header['Ngroups_ThisFile'])

    fileinst += 1
  else:
    print('Warning: halo %d not found'%subhalo_number)

  return prog, desc, pos, vel, mass, ID, header

def trace_subhalo(snapshot_number,subhalo_number):

  '''
  
  Trace a subhalo's position, mass, and other tracking information across snapshots.
  
  Parameters:
    
    snapshot_number

    subhalo_number
    
  Returns:
    
    num: snapshot number

    time: scale factor array, shape (NT,)

    pos: position array, shape (NT,3), comoving
    
    vel: velocity array, shape (NT,3), peculiar
    
    mass: mass array, shape (NT,)

    group: host group, shape (NT,)

    ID: list of dicts with group ID, subhalo ID, and most bound particle ID; shape (NT,)
    
  '''

  prog, desc, pos_, vel_, mass_, ID_, header_ = subhalo_tracing_data(snapshot_number,subhalo_number)

  print('halo: %d in snapshot %d'%(subhalo_number,snapshot_number))

  pos = [pos_]
  vel = [vel_]
  mass = [mass_]
  ID = [ID_]
  time = [header_['Time']]
  num = [snapshot_number]

  shift = 0
  while prog >= 0:
    shift += 1
    print('progenitor: %d in snapshot %d'%(prog,snapshot_number-shift))

    prog, _, pos_, vel_, mass_, ID_, header_ = subhalo_tracing_data(snapshot_number-shift,prog)

    pos += [pos_]
    vel += [vel_]
    mass += [mass_]
    ID += [ID_]
    time += [header_['Time']]
    num += [snapshot_number-shift]

  pos = pos[::-1]
  vel = vel[::-1]
  mass = mass[::-1]
  ID = ID[::-1]
  time = time[::-1]
  num = num[::-1]

  shift = 0
  while desc >= 0:
    shift += 1
    print('descendant: %d in snapshot %d'%(desc,snapshot_number+shift))

    _, desc, pos_, vel_, mass_, ID_, header_ = subhalo_tracing_data(snapshot_number+shift,desc)

    pos += [pos_]
    vel += [vel_]
    mass += [mass_]
    ID += [ID_]
    time += [header_['Time']]
    num += [snapshot_number+shift]

  return np.array(num), np.array(time), np.array(pos), np.array(vel), mass, ID

def subhalo_group_data(fileprefix,opts={'mass':True,'len':False,'pos':False},parttype=None):

  '''
  
  Read halos from GADGET HDF5 FOF+subhalo file and return data relevant to group membership.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_subhalo_tab_000, not fof_subhalo_tab_000.0.hdf5)

    opts: which fields to read and return

    parttype: if not None, consider only particles of the given type for certain outputs

  Returns:
    
    group: host group number

    rank: rank of subhalo within host group

    parentrank: rank of parent subhalo within host group

    mass: subhalo mass

    groupmass: mass of host group

    length: subhalo particle count

    grouplength: host group particle count

    pos: subhalo position (NH,3)

    grouppos: host group position (NH,3)

    header: a dict with header info, use list(header) to see the fields
    
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  fileinst = 0
  group = []
  rank = []
  parentrank = []
  if opts.get('mass'):
    mass = []
    _groupmass = []
  if opts.get('len'):
    length = []
    _grouplength = []
  if opts.get('pos'):
    pos = []
    _grouppos = []
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

      group += [np.array(f['Subhalo/SubhaloGroupNr'])]
      rank += [np.array(f['Subhalo/SubhaloRankInGr'])]
      parentrank += [np.array(f['Subhalo/SubhaloParentRank'])]
      if parttype is None:
        if opts.get('mass'):
          mass += [np.array(f['Subhalo/SubhaloMass'])]
          _groupmass += [np.array(f['Group/GroupMass'])]
        if opts.get('len'):
          length += [np.array(f['Subhalo/SubhaloLen'])]
          _grouplength += [np.array(f['Group/GroupLen'])]
      else:
        if opts.get('mass'):
          mass += [np.array(f['Subhalo/SubhaloMassType'][:,parttype])]
          _groupmass += [np.array(f['Group/GroupMassType'][:,parttype])]
        if opts.get('len'):
          length += [np.array(f['Subhalo/SubhaloLenType'][:,parttype])]
          _grouplength += [np.array(f['Group/GroupLenType'][:,parttype])]
      if opts.get('pos'):
        pos += [np.array(f['Subhalo/SubhaloPos'])]
        _grouppos += [np.array(f['Group/GroupPos'])]

    fileinst += 1

  group = np.concatenate(group)

  ret = [group, np.concatenate(rank), np.concatenate(parentrank)]

  if opts.get('mass'):
    groupmass = np.concatenate(_groupmass)[group]
    ret += [np.concatenate(mass), groupmass]
  if opts.get('len'):
    grouplength = np.concatenate(_grouplength)[group]
    ret += [np.concatenate(length), grouplength]
  if opts.get('pos'):
    grouppos = np.concatenate(_grouppos,axis=0)[group]
    ret += [np.concatenate(pos), grouppos]

  return tuple(ret + [header])

def group_extent(fileprefix,group,size_definition='TopHat200'):

  '''
  
  Return position and extent of a group from a GADGET HDF5 FOF file (SUBFIND required).
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_tab_000, not fof_tab_000.0.hdf5)

    group: group number

    size_definition: 'Crit200', 'Crit500', 'Mean200', or 'TopHat200' (default)
    
  Returns:
    
    pos: shape (3,), comoving
    
    radius
    
    header: a dict with header info, use list(header) to see the fields
    
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  fileinst = 0
  hinst = 0
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      numfiles = header['NumFiles']

      if hinst + header['Ngroups_ThisFile'] > group:
        index = group - hinst

        pos = np.array(f['Group/GroupPos'])[index]
        radius = np.array(f['Group/Group_R_'+size_definition])[index]

        break

      hinst += int(header['Ngroups_ThisFile'])

    fileinst += 1
  else:
    print('Warning: halo %d not found'%group)
    pos = np.zeros(3)
    radius = 0

  return pos, radius, header

def group_data(fileprefix,group,size_definition='Mean200',opts={'pos':True,'vel':False,'radius':True,'mass':True,}):

  '''
  
  Return position, radius, mass of a group from a GADGET HDF5 FOF file (SUBFIND required).
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_tab_000, not fof_tab_000.0.hdf5)

    group: group number

    size_definition: 'Crit200', 'Crit500', 'Mean200' (default), or 'TopHat200'

    opts: which fields to return
    
  Returns:
    
    pos: shape (3,), comoving

    vel: shape (3,), peculiar
    
    radius

    mass
    
    header: a dict with header info, use list(header) to see the fields
    
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  fileinst = 0
  hinst = 0
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

      if hinst + header['Ngroups_ThisFile'] > group:
        index = group - hinst

        pos = np.array(f['Group/GroupPos'])[index]
        vel = np.array(f['Group/GroupVel'])[index] * np.sqrt(ScaleFactor)
        radius = np.array(f['Group/Group_R_'+size_definition])[index]
        mass = np.array(f['Group/Group_M_'+size_definition])[index]

        break

      hinst += int(header['Ngroups_ThisFile'])

    fileinst += 1
  else:
    print('Warning: halo %d not found'%group)
    pos = np.zeros(3)
    vel = np.zeros(3)
    radius = 0.
    mass = 0.

  ret = []
  if opts.get('pos'): ret += [pos]
  if opts.get('vel'): ret += [vel]
  if opts.get('radius'): ret += [radius]
  if opts.get('mass'): ret += [mass]
  ret += [header]

  return tuple(ret)

def count_halos():

  '''
  
  Trace a subhalo's position, mass, and other tracking information across snapshots.
  
  Parameters:
    
  Returns:
    
    num: snapshot number

    times: scale factor array, shape (NT,)

    groups: group count array, shape (NT,)

    subhalos: subhalo count array, shape (NT,)
    
    mass: total mass in halos, shape (NT,)

  '''

  times = []
  groups = []
  subhalos = []
  mass = []

  snapshot_number = 0
  while True:
    fileprefix = fileprefix_subhalo%(snapshot_number,snapshot_number)
    try:
      mass_, header = fof_to_halos(fileprefix,opts={'pos':False,'vel':False,'mass':True})
    except Exception as e:
      print(e)
      break

    times += [header['Time']]
    groups += [header['Ngroups_Total']]
    subhalos += [header['Nsubhalos_Total']]
    mass += [np.sum(mass_)]

    snapshot_number += 1

  return np.arange(snapshot_number), np.array(times), np.array(groups), np.array(subhalos), np.array(mass)

def list_snapshots():

  '''
  
  Get all snapshot names and headers.
  
  Parameters:
    
  Returns:
    
    names: list of snapshot prefixes

    headers: list of header dicts
    
  '''

  snapshot_number = 0

  names = []
  headers = []

  while True:
    prefix = fileprefix_snapshot%(snapshot_number,snapshot_number)

    filepath = [
      Path(prefix + file_extension),
      Path(prefix + '.0' + file_extension),
      Path(prefix.split('/')[-1] + file_extension),
      ]

    if filepath[0].is_file():
      filename = prefix + file_extension
      numfiles = 1
      names += [prefix]
    elif filepath[1].is_file():
      filename = prefix + '.0' + file_extension
      numfiles = 2
      names += [prefix]
    elif filepath[2].is_file():
      filename = prefix.split('/')[-1] + file_extension
      numfiles = 1
      names += [prefix.split('/')[-1]]
    else:
      break


    with h5py.File(filename, 'r') as f:
      headers += [dict(f['Header'].attrs)]

    snapshot_number += 1

  return names, headers

def read_particles_filter(fileprefix, center=None, rotation=None, radius=None, halfwidth=None, ID_list=None, type_list=None, part_range=None, opts={'pos':True,'vel':True,'ID':False,'mass':True,'index':False,'type':False,'acc':False,'pot':False,'density':False,'energy':False},chunksize=1048576,verbose=True):

  '''
  
  Read particles from GADGET HDF5 snapshot. Return only particles matching filter.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)

    center: center particles about this position

    rotation: 3x3 rotation matrix to apply. Requires center.

    radius: filter particles within radius. Requires center. If length 2, specifies (min,max).

    halfwidth: filter particles within box of width 2*halfwidth. Requires center.

    ID_list: only include particles whose IDs are listed

    type_list: only include particles of the listed types

    part_range: only read particles whose positions within the files lie within range.
      Useful with group-ordered snapshots.
      Format: ([minType0,minType1,...], [maxType0+1,maxType1+1,...])

    opts: which fields to read and return

    chunksize: restrict number of particles to handle at a time (to save memory).
      Make as large as possible if using ID_list.
    
  Returns:
    
    pos: position array, shape (NP,3), comoving
    
    vel: velocity array, shape (NP,3), peculiar

    ID: ID array, shape (NP,)
    
    mass: mass array, shape (NP,)

    index: array of particle indices within snapshot, shape (NP,)

    type: array of particle types, shape (NP,)

    acc: acceleration, shape (NP,3)

    pot: potential, shape (NP,)

    density: density of gas, shape (NP,)

    energy: internal energy of gas, shape (NP,)
    
    header: a dict with header info, use list(header) to see the fields
  
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  if center is not None:
    if len(np.shape(center)) < 2:
      center = np.array(center)[None,:]
    else:
      center = np.array(center)

  try:
    rmin = radius[0]
    rmax = radius[1]
  except:
    rmin = None
    rmax = radius

  fileinst = 0
  if opts.get('pos'): pos = []
  if opts.get('vel'): vel = []
  if opts.get('mass'): mass = []
  if opts.get('ID'): ID = []
  if opts.get('index'): index = []
  if opts.get('type'): ptype = []
  if opts.get('acc'): acc = []
  if opts.get('pot'): pot = []
  if opts.get('density'): density = []
  if opts.get('energy'): energy = []
  header = None
  while fileinst < numfiles:

    if header is not None and ID_list is not None and len(ID_list) == 0:
      break

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      if verbose: print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      MassTable = header['MassTable']
      ScaleFactor = 1./(1+header['Redshift'])
      NP = header['NumPart_ThisFile']
      NPtot = header['NumPart_Total']
      numfiles = header['NumFilesPerSnapshot']
      BoxSize = header['BoxSize']
      if 'PERIODIC' not in dict(f['Config'].attrs):
        BoxSize = 0.

      if fileinst == 0:
        if type_list is None:
          types = np.arange(len(NPtot))
        else:
          types = np.intersect1d(type_list,np.arange(len(NPtot)))
          if verbose: print('  types ' + ', '.join([str(x) for x in types]))
        i00 = np.zeros(len(NPtot),dtype=np.int64)

      nreadTot = 0
      for typ in types:
        NPtyp = int(NP[typ])
        if NPtyp == 0:
          continue

        Nleft = NPtyp
        i0 = 0

        while Nleft > 0:
          Nc = min(chunksize,Nleft)
          iread = slice(i0,i0+Nc)

          idx = np.full(Nc,True)

          if center is not None: # filter by position
            pos_ = np.array(f['PartType%d/Coordinates'%typ][iread])
            pos_ -= center
            if BoxSize > 0.:
              pos_[pos_ > .5*BoxSize] -= BoxSize
              pos_[pos_ < -.5*BoxSize] += BoxSize
            if rotation is not None: pos_ = (rotation@(pos_.T)).T
            if rmax is not None:
              idx &= np.sum(pos_**2,axis=1) < rmax**2
            if rmin is not None:
              idx &= np.sum(pos_**2,axis=1) > rmin**2
            if halfwidth is not None:
              idx &= (-halfwidth<=pos_[:,0])&(pos_[:,0]<=halfwidth)
              idx &= (-halfwidth<=pos_[:,1])&(pos_[:,1]<=halfwidth)
              idx &= (-halfwidth<=pos_[:,2])&(pos_[:,2]<=halfwidth)

          if part_range is not None: # filter by position in file
            idx &= part_range[0][typ] <= np.arange(i00[typ],i00[typ]+Nc)
            idx &= np.arange(i00[typ],i00[typ]+Nc) < part_range[1][typ]

          if ID_list is not None: # filter by ID
            # ID_ = np.array(f['PartType%d/ParticleIDs'%typ][iread])
            # idx &= np.isin(ID_,ID_list,assume_unique=True)
            # this can be slow, so save time by doing this test
            # only on particles that passed other tests
            ID_ = np.array(f['PartType%d/ParticleIDs'%typ][iread])[idx]
            #print('    testing if %d IDs in list of %d'%(len(ID_),len(ID_list)))
            idx_ID = np.isin(ID_,ID_list,assume_unique=True)
            ID_list = np.array(ID_list)[np.isin(ID_list,ID_[idx_ID],assume_unique=True,invert=True)]
            idx[idx] &= idx_ID

          nread = np.sum(idx)
          nreadTot += nread
          if nread > 0:
            if opts.get('pos'):
              if center is None:
                pos += [np.array(f['PartType%d/Coordinates'%typ][iread])[idx]]
              else: pos += [pos_[idx]]
            if opts.get('vel'):
              vel += [np.array(f['PartType%d/Velocities'%typ][iread])[idx] * np.sqrt(ScaleFactor)]

            if opts.get('mass'):
              if MassTable[typ] == 0.:
                mass_ = np.array(f['PartType%d/Masses'%typ][iread])
              else:
                mass_ = np.full(Nc,MassTable[typ])
              mass += [mass_[idx]]
            if opts.get('ID'):
              if ID_list is None:
                ID += [np.array(f['PartType%d/ParticleIDs'%typ][iread])[idx]]
              else: ID += [ID_[idx_ID]]
            if opts.get('index'):
              index += [np.arange(i00[typ],i00[typ]+Nc,dtype=np.int64)[idx]]
            if opts.get('type'):
              ptype += [np.full(np.sum(idx),typ)]

            if opts.get('acc'):
              acc += [np.array(f['PartType%d/Acceleration'%typ][iread])[idx] * np.sqrt(ScaleFactor)]
            if opts.get('pot'):
              pot += [np.array(f['PartType%d/Potential'%typ][iread])[idx]]
            if opts.get('density'):
              density += [np.array(f['PartType%d/Density'%typ][iread])[idx]]
            if opts.get('energy'):
              energy += [np.array(f['PartType%d/InternalEnergy'%typ][iread])[idx]]

          Nleft -= Nc
          i0 += Nc
          i00[typ] += Nc

    if verbose: print('  ' + str(nreadTot) + ' particles match filter')
    fileinst += 1

  ret = []
  if opts.get('pos'): ret += [np.concatenate(pos,axis=0)] if len(pos) > 0 else [np.empty((0,3))]
  if opts.get('vel'): ret += [np.concatenate(vel,axis=0)] if len(vel) > 0 else [np.empty((0,3))]
  if opts.get('mass'): ret += [np.concatenate(mass)] if len(mass) > 0 else [np.empty(0)]
  if opts.get('ID'): ret += [np.concatenate(ID)] if len(ID) > 0 else [np.empty(0,dtype=np.uint32)]
  if opts.get('index'): ret += [np.concatenate(index)] if len(index) > 0 else [np.empty(0,dtype=np.uint32)]
  if opts.get('type'): ret += [np.concatenate(ptype)] if len(ptype) > 0 else [np.empty(0,dtype=np.uint32)]
  if opts.get('acc'): ret += [np.concatenate(acc,axis=0)] if len(acc) > 0 else [np.empty((0,3))]
  if opts.get('pot'): ret += [np.concatenate(pot)] if len(pot) > 0 else [np.empty(0)]
  if opts.get('density'): ret += [np.concatenate(density)] if len(density) > 0 else [np.empty(0)]
  if opts.get('energy'): ret += [np.concatenate(energy)] if len(energy) > 0 else [np.empty(0)]
  ret += [header]

  return tuple(ret)

def read_subhalos(fileprefix, opts={'pos':True,'vel':True,'mass':True,'radius':False,'lentype':False,'group':False},group_opts={},sodef='Mean200',verbose=True):

  '''
  
  Read subhalos from SUBFIND catalogue. Also read groups.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)

    opts: which fields to read and return

    group_opts: for groups, which fields to read and return

    sodef: spherical overdensity definition for groups, default 'Mean200'
    
  Returns:
    
    pos: position array, shape (NH,3), comoving
    
    vel: velocity array, shape (NH,3), peculiar

    mass: mass array, shape (NH,)

    radius: half-mass radius, shape (NH,)

    veldisp: velocity dispersion, shape (NH,)

    lentype: particle count array (by type), shape (NH,Ntype)
    
    group: subhalo's group, shape (NH,)

    rank: subhalo rank in group

    parentrank: rank of parent subhalo within group

    mostbound: ID of most bound particle

    group_pos: position array, shape (NG,3), comoving

    group_vel: 

    group_mass: FOF mass

    group_somass: SO mass
  
    group_soradius: SO radius

    group_lentype: 

    group_firstsub: 

    group_numsubs: 
    
    header: a dict with header info, use list(header) to see the fields
  
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  fileinst = 0
  if opts.get('pos'): pos = []
  if opts.get('vel'): vel = []
  if opts.get('mass'): mass = []
  if opts.get('radius'): radius = []
  if opts.get('veldisp'): veldisp = []
  if opts.get('lentype'): lentype = []
  if opts.get('group'): group = []
  if opts.get('rank'): rank = []
  if opts.get('parentrank'): parentrank = []
  if opts.get('mostbound'): mostbound = []
  if group_opts.get('pos'): group_pos = []
  if group_opts.get('vel'): group_vel = []
  if group_opts.get('mass'): group_mass = []
  if group_opts.get('somass'): group_somass = []
  if group_opts.get('soradius'): group_soradius = []
  if group_opts.get('lentype'): group_lentype = []
  if group_opts.get('firstsub'): group_firstsub = []
  if group_opts.get('numsubs'): group_numsubs = []
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst


    with h5py.File(filename, 'r') as f:
      if verbose: print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      ScaleFactor = 1./(1+header['Redshift'])
      numfiles = header['NumFiles']
      BoxSize = header['BoxSize']

      try:
        if opts.get('pos'):
          pos += [np.array(f['Subhalo/SubhaloPos'])]
        if opts.get('vel'):
          vel += [np.array(f['Subhalo/SubhaloVel']) * np.sqrt(ScaleFactor)]
        if opts.get('mass'):
          mass += [np.array(f['Subhalo/SubhaloMass'])]
        if opts.get('radius'):
          radius += [np.array(f['Subhalo/SubhaloHalfmassRad'])]
        if opts.get('veldisp'):
          veldisp += [np.array(f['Subhalo/SubhaloVelDisp'])]
        if opts.get('lentype'):
          lentype += [np.array(f['Subhalo/SubhaloLenType'])]
        if opts.get('group'):
          group += [np.array(f['Subhalo/SubhaloGroupNr'])]
        if opts.get('rank'):
          rank += [np.array(f['Subhalo/SubhaloRankInGr'])]
        if opts.get('parentrank'):
          parentrank += [np.array(f['Subhalo/SubhaloParentRank'])]
        if opts.get('mostbound'):
          mostbound += [np.array(f['Subhalo/SubhaloIDMostbound'])]
      except KeyError as e:
        pass
      try:
        if group_opts.get('pos'):
          group_pos += [np.array(f['Group/GroupPos'])]
        if group_opts.get('vel'):
          group_vel += [np.array(f['Group/GroupVel']) * np.sqrt(ScaleFactor)]
        if group_opts.get('mass'):
          group_mass += [np.array(f['Group/GroupMass'])]
        if group_opts.get('somass'):
          group_somass += [np.array(f['Group/Group_M_%s'%sodef])]
        if group_opts.get('soradius'):
          group_soradius += [np.array(f['Group/Group_R_%s'%sodef])]
        if group_opts.get('lentype'):
          group_lentype += [np.array(f['Group/GroupLenType'])]
        if group_opts.get('firstsub'):
          group_firstsub += [np.array(f['Group/GroupFirstSub'])]
        if group_opts.get('numsubs'):
          group_numsubs += [np.array(f['Group/GroupNsubs'])]
      except KeyError as e:
        pass

    fileinst += 1

  ret = []
  if opts.get('pos'): ret += [np.concatenate(pos,axis=0)]
  if opts.get('vel'): ret += [np.concatenate(vel,axis=0)]
  if opts.get('mass'): ret += [np.concatenate(mass)]
  if opts.get('radius'): ret += [np.concatenate(radius)]
  if opts.get('veldisp'): ret += [np.concatenate(veldisp)]
  if opts.get('lentype'): ret += [np.concatenate(lentype,axis=0)]
  if opts.get('group'): ret += [np.concatenate(group)]
  if opts.get('rank'): ret += [np.concatenate(rank)]
  if opts.get('parentrank'): ret += [np.concatenate(parentrank)]
  if opts.get('mostbound'): ret += [np.concatenate(mostbound)]
  if group_opts.get('pos'): ret += [np.concatenate(group_pos,axis=0)]
  if group_opts.get('vel'): ret += [np.concatenate(group_vel,axis=0)]
  if group_opts.get('mass'): ret += [np.concatenate(group_mass)]
  if group_opts.get('somass'): ret += [np.concatenate(group_somass)]
  if group_opts.get('soradius'): ret += [np.concatenate(group_soradius)]
  if group_opts.get('lentype'): ret += [np.concatenate(group_lentype,axis=0)]
  if group_opts.get('firstsub'): ret += [np.concatenate(group_firstsub)]
  if group_opts.get('numsubs'): ret += [np.concatenate(group_numsubs)]
  ret += [header]

  return tuple(ret)

def read_params(fileprefix):

  '''
  
  Read params from any HDF5 output.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)

  Returns:
    
    params: a dict with params

  
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  if numfiles == 1:
    filename = filebase
  else:
    filename = filebase%0

  with h5py.File(filename, 'r') as f:
    params = dict(f['Parameters'].attrs)

  return params

def read_header(fileprefix):

  '''
  
  Read params from any HDF5 output.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)

  Returns:
    
    header

  
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  if numfiles == 1:
    filename = filebase
  else:
    filename = filebase%0

  with h5py.File(filename, 'r') as f:
    header = dict(f['Header'].attrs)

  return header

def particles_by_ID(fileprefix, ID_list, opts, chunksize=1048576):

  '''
  
  Read specific particles from GADGET HDF5 snapshot and return data in the same shape.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)

    ID_list: particle IDs (any shape)

    opts: which fields to read and return

    chunksize: restrict number of particles to handle at a time (to save memory).
    
  Returns:

    pos: comoving position (shape ID_list.shape + (3,))

    vel: particle types (shape ID_list.shape + (3,))
    
    mass: particle types (shape ID_list.shape)

    index: particle indices within snapshot (same shape as ID_list)

    type: particle types (same shape as ID_list)

    header: a dict with header info, use list(header) to see the fields
  
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  assume_unique = np.unique(ID_list).size == ID_list.size

  fileinst = 0
  ID_list = np.array(ID_list)
  index = np.zeros(np.shape(ID_list),dtype=np.int64)-1
  if opts.get('pos'): pos = np.zeros(np.shape(ID_list) + (3,)) * np.nan
  if opts.get('vel'): vel = np.zeros(np.shape(ID_list) + (3,)) * np.nan
  if opts.get('mass'): mass = np.zeros(np.shape(ID_list)) * np.nan
  if opts.get('type'): ptype = np.zeros(np.shape(ID_list),dtype=np.int32)-1
  header = None
  while fileinst < numfiles:

    if np.all(index>=0):
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
        types = np.arange(len(NPtot))
        i00 = np.zeros(len(NPtot),dtype=np.int64)

      for typ in types:
        NPtyp = int(NP[typ])
        if NPtyp == 0:
          continue

        Nleft = NPtyp
        i0 = 0

        while Nleft > 0:
          Nc = min(chunksize,Nleft)
          iread = slice(i0,i0+Nc)

          ID = np.array(f['PartType%d/ParticleIDs'%typ][iread])

          in_idx = np.isin(ID_list,ID,assume_unique=assume_unique) # shape ID_list.shape

          if np.any(in_idx):
            sort_idx = ID.argsort()
            test = np.searchsorted(ID,ID_list[in_idx],sorter=sort_idx)
            index[in_idx] = sort_idx[test] + i00[typ]
            if opts.get('pos'):
              pos[in_idx] = np.array(f['PartType%d/Coordinates'%typ][iread])[sort_idx[test]]
            if opts.get('vel'):
              vel[in_idx] = np.array(f['PartType%d/Velocities'%typ][iread])[sort_idx[test]] * np.sqrt(ScaleFactor)
            if opts.get('mass'):
              if MassTable[typ] == 0.:
                mass[in_idx] = np.array(f['PartType%d/Masses'%typ][iread])[sort_idx[test]]
              else:
                mass[in_idx] = MassTable[typ]
            if opts.get('type'):
              ptype[in_idx] = typ

          Nleft -= Nc
          i0 += Nc
          i00[typ] += Nc

    fileinst += 1

  ret = []
  if opts.get('pos'): ret += [pos]
  if opts.get('vel'): ret += [vel]
  if opts.get('mass'): ret += [mass]
  if opts.get('index'): ret += [index]
  if opts.get('type'): ret += [ptype]
  ret += [header]

  return tuple(ret)

def particle_subhalo_data(fileprefix,index,ptype):

  '''

  UNFINISHED
  
  Get subhalo/group membership data for particles.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_subhalo_tab_000, not fof_subhalo_tab_000.0.hdf5)

    index: SORTED particle indices within corresponding snapshot

    ptype: particle types within corresponding snapshot

  Returns:
    
    subhalo_number

    subhalo_mass

    group_number

    group_mass
    
  '''

  filebase, numfiles = _get_filebase(fileprefix)

  fileinst = 0
  subhalonum = []
  subhalomass = []
  groupnum = []
  groupmass = []
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

      group += [np.array(f['Subhalo/SubhaloGroupNr'])]
      rank += [np.array(f['Subhalo/SubhaloRankInGr'])]
      parentrank += [np.array(f['Subhalo/SubhaloParentRank'])]
      if parttype is None:
        if opts.get('mass'):
          mass += [np.array(f['Subhalo/SubhaloMass'])]
          _groupmass += [np.array(f['Group/GroupMass'])]
        if opts.get('len'):
          length += [np.array(f['Subhalo/SubhaloLen'])]
          _grouplength += [np.array(f['Group/GroupLen'])]
      else:
        if opts.get('mass'):
          mass += [np.array(f['Subhalo/SubhaloMassType'][:,parttype])]
          _groupmass += [np.array(f['Group/GroupMassType'][:,parttype])]
        if opts.get('len'):
          length += [np.array(f['Subhalo/SubhaloLenType'][:,parttype])]
          _grouplength += [np.array(f['Group/GroupLenType'][:,parttype])]
      if opts.get('pos'):
        pos += [np.array(f['Subhalo/SubhaloPos'])]
        _grouppos += [np.array(f['Group/GroupPos'])]

    fileinst += 1

  group = np.concatenate(group)

  ret = [group, np.concatenate(rank), np.concatenate(parentrank)]

  if opts.get('mass'):
    groupmass = np.concatenate(_groupmass)[group]
    ret += [np.concatenate(mass), groupmass]
  if opts.get('len'):
    grouplength = np.concatenate(_grouplength)[group]
    ret += [np.concatenate(length), grouplength]
  if opts.get('pos'):
    grouppos = np.concatenate(_grouppos,axis=0)[group]
    ret += [np.concatenate(pos), grouppos]

  return tuple(ret + [header])

def subhalo_desc(snapshot_number,subhalo_numbers):

  '''
  
  For a list of subhalo numbers, get the list of descendants.
  
  Parameters:
    
    snapshot_number

    subhalo_numbers
    
  Returns:
    
    desc: best-scoring descendant subhalo
  
    header: a dict with header info, use list(header) to see the fields
    
  '''

  prefix_desc = fileprefix_subhalo_desc%(snapshot_number,snapshot_number)

  filebase, numfiles = _get_filebase(prefix_desc)

  subhalo_numbers = np.array(subhalo_numbers)
  desc = np.full_like(subhalo_numbers,-1)

  fileinst = 0
  hinst = 0
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)
      numfiles = header['NumFiles']

      if header['Nsubhalos_ThisFile'] > 0:
        # get best-matching descendent (decided already by SUBFIND)
        idx = subhalo_numbers-hinst
        valid = (0<idx)&(idx<header['Nsubhalos_ThisFile'])
        desc[valid] = np.array(f['Subhalo/DescSubhaloNr'])[idx[valid]]

      # advance halo indices as we advance in file number
      hinst += int(header['Nsubhalos_ThisFile'])

    fileinst += 1

  return desc, header

