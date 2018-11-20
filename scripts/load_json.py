import json
import sys
import numpy as np

berts = []
with open(sys.argv[1], 'r') as fin:
  line = fin.readline()
  while line:
    bert = json.loads(line)
    line = fin.readline()
    b = np.array(bert['features'][0]['layers'][0]['values'])
    berts.append(b)
    print bert['features'][0]['layers'][0]['values'][:4]
  print (berts[0] + berts[1])[:4]

