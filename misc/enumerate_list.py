
def run(argv):
  try:
    infile = argv[1]
    outfile = argv[2]
  except:
    print('python script.py <in-file> <out-file>')
    return

  with open(infile,'r') as fin:
    with open(outfile,'w') as fout:
      for i,line in enumerate(fin):
        fout.write('%d '%i + line)
  return

if __name__ == '__main__':
  from sys import argv
  run(argv)

