import numpy as np

def run(argv):

  if len(argv) < 3:
    print('python script.py <ID-val.npz> <threshold> [above=0] [out-file]')
    return 1
  
  data = np.load(argv[1])
  ID_ = data['ID']
  val_ = data[list(data)[list(data) != 'ID'][0]]

  threshold = float(argv[2])

  outfile = argv[1][:-4] + '_%g.bin'%threshold
  if len(argv) > 4:
    outfile = argv[4]

  if len(argv) > 3 and argv[3].lower().startswith(['t','y','1']):
    print(' > %g'%threshold)
    idx = val_ > threshold
  else:
    print(' < %g'%threshold)
    idx = val_ < threshold

  ID = ID_[idx].astype(np.uint32)

  ID.tofile(outfile)

  print('to %s: %d IDs (of %d)'%(outfile,ID.size,ID_.size))
  return

if __name__ == '__main__':
  from sys import argv
  run(argv)
