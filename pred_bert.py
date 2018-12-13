from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import codecs
import re
import numpy as np
import json
import argparse
from list2bert import FLAGS
from list2bert import get_bert

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

def list_to_bert(sents, bert_file, layer, bert_vocab, bert_config, bert_model, max_seq=256,batch_size=8):
  flags = FLAGS()
  flags.bert_config_file = bert_config
  flags.vocab_file = bert_vocab
  flags.init_checkpoint = bert_model
  flags.output_file = bert_file
  flags.layers = layer
  flags.max_seq_length = max_seq
  flags.batch_size = batch_size
  get_bert(sents, flags)
  
def merge(bert_file, merge_file, sents):
  n = 0
  n_unk = 0
  n_tok = 0
  fo = codecs.open(merge_file, 'w')
  with codecs.open(bert_file, 'r') as fin:
    line = fin.readline()
    while line:
      if n % 100 == 0:
        print ("\r%d" % n, end='')
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
              print (sents[n], len(merged["features"]))
            else:
              merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        elif item["token"] == "[UNK]":
          n_unk += 1
          merged["features"].append(item)
          if len(sents[n]) < len(merged["features"]) - 1:
            print (sents[n], len(merged["features"]))
          else:
            merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        else:
          merged["features"].append(item)
      try:
        assert len(merged["features"]) == len(sents[n]) + 2
      except:
        orig = [m["token"] for m in merged["features"]]
        print ('\n',len(merged["features"]), len(sents[n]))
        print (sents[n], '\n', orig)
        print (zip(sents[n], orig[1:-1]))
        raise ValueError("Sentence-{}:{}".format(n, ' '.join(sents[n])))
      for i in range(len(sents[n])):
        try:
          assert sents[n][i].lower() == merged["features"][i+1]["token"]
        except:
          print ('wrong word id:{}, word:{}'.format(i, sents[n][i]))
      n_tok += len(sents[n])
      fo.write(json.dumps(merged).encode('utf-8')+"\n")
      line = fin.readline()
      n += 1
    print ('Total tokens:{}, UNK tokens:{}'.format(n_tok, n_unk))
    info_file=os.path.dirname(merge_file) + '/README.txt'
    print (info_file)
    with open(info_file, 'a') as info:
      info.write('File:{}\nTotal tokens:{}, UNK tokens:{}\n\n'.format(merge_file, n_tok, n_unk))

parser = argparse.ArgumentParser()
parser.add_argument('conll_file', type=str, help='input conllu file')
parser.add_argument('bert_file', type=str, help='output bert file')
parser.add_argument('merge_file', type=str, help='output merged bert file')
parser.add_argument('--bert_vocab', type=str, help='bert vocab')
parser.add_argument('--bert_config', type=str, help='bert config')
parser.add_argument('--bert_model', type=str, help='bert checkpoint')
parser.add_argument('--layer', type=str, default='-1', help='bert layers')
#parser.add_argument('--gpu', type=str, help='GPU')
args = parser.parse_args()

n = 0
sents = []
for sent in load_conllu(args.conll_file):
  sents.append(sent)
print ("Total {} Sentences".format(len(sents)))
list_to_bert(sents,args.bert_file,args.layer,args.bert_vocab,args.bert_config,args.bert_model,max_seq=512)
merge(args.bert_file, args.merge_file, sents)
