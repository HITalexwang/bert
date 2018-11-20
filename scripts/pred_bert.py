import sys, os
import codecs
import re
import numpy as np
import json

def load_conllu(file):
  with codecs.open(file, 'rb') as f:
    reader = codecs.getreader('utf-8')(f)
    buff = []
    for line in reader:
      line = line.strip()
      if line and not line.startswith('#'):
        if not re.match('[0-9]+[-.][0-9]+', line):
          buff.append(line.split('\t')[1])
      elif buff:
        yield buff
        buff = []
    if buff:
      yield buff

def to_raw(sents, file):
  with codecs.open(file, 'w') as fo:
    for sent in sents:
      fo.write((" ".join(sent)).encode('utf-8')+'\n')

def bert(raw_file, bert_file, gpu, layer, model, max_seq='512',batch_size='8'):
  dict = {'gpu':gpu, 'input':raw_file, 'output':bert_file, 'layer':layer, 'BERT_BASE_DIR':model, 'max_seq':max_seq, 'batch_size':batch_size}
  cmd = "source /users2/yxwang/work/env/py2.7/bin/activate ; CUDA_VISIBLE_DEVICES={gpu} python extract_features.py --input_file={input} --output_file={output} --vocab_file={BERT_BASE_DIR}/vocab.txt --bert_config_file={BERT_BASE_DIR}/bert_config.json --init_checkpoint={BERT_BASE_DIR}/bert_model.ckpt --layers={layer} --max_seq_length={max_seq} --batch_size={batch_size}".format(**dict)
  #cmd = "./scripts/extract.sh {gpu} {input} {output} {layer}".format(**dict)
  print cmd
  os.system(cmd)

def merge(bert_file, merge_file, sents):
  n = 0
  fo = codecs.open(merge_file, 'w')
  with codecs.open(bert_file, 'r') as fin:
    line = fin.readline()
    while line:
      if n % 100 == 0:
        print >> sys.stderr, "\r%d" % n,
      bert = json.loads(line)
      tokens = []
      merged = {"linex_index": bert["linex_index"], "features":[]}
      for i, item in enumerate(bert["features"]):
        if item["token"]=="[CLS]" or item["token"]=="[SEP]":
          merged["features"].append(item)
          continue
        if item["token"].startswith("##") and not (len(merged["features"])-1<len(sents[n]) and item["token"] == sents[n][len(merged["features"])-1]):
          for j, layer in enumerate(merged["features"][-1]["layers"]):
            merged["features"][-1]["layers"][j]["values"] = list(np.array(layer["values"]) + np.array(item["layers"][j]["values"]))
            if len(sents[n]) < len(merged["features"]) - 1:
              print sents[n], len(merged["features"])
            else:
              merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        elif item["token"] == "[UNK]":
          merged["features"].append(item)
          if len(sents[n]) < len(merged["features"]) - 1:
            print sents[n], len(merged["features"])
          else:
            merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        else:
          merged["features"].append(item)
      try:
        assert len(merged["features"]) == len(sents[n]) + 2
      except:
        orig = [m["token"] for m in merged["features"]]
        print '\n',len(merged["features"]), len(sents[n])
        print sents[n], '\n', orig
        print zip(sents[n], orig[1:-1])
        raise ValueError("Sentence-{}:{}".format(n, ' '.join(sents[n])))
      for i in range(len(sents[n])):
        try:
          assert sents[n][i].lower() == merged["features"][i+1]["token"]
        except:
          print 'wrong word id:{}, word:{}'.format(i, sents[n][i])

      fo.write(json.dumps(merged).encode('utf-8')+"\n")
      line = fin.readline()
      n += 1

if len(sys.argv) < 8:
  print "usage:%s [gpu] [model] [layer(-1,-2,-3,-4)] [conllu file] [output raw] [output bert] [merged bert]" % sys.argv[0]
  exit(1)

gpu = sys.argv[1]
model = sys.argv[2]
layer = sys.argv[3]
conll_file = sys.argv[4]
raw_file = sys.argv[5]
bert_file = sys.argv[6]
merge_file = sys.argv[7]

n = 0
sents = []
for sent in load_conllu(conll_file):
  sents.append(sent)
print "Total {} Sentences".format(len(sents))
to_raw(sents, raw_file)
bert(raw_file,bert_file,gpu,layer,model)
merge(bert_file, merge_file, sents)
