# -*- coding: utf-8 -*-
import os

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

print "Usage: fja fen fhf prefix label_num"
fja = sys.argv[1]
fen = sys.argv[2]
fhf = sys.argv[3]
prefix = sys.argv[4]
ln = sys.argv[5]

import codecs

def read_all_lines(fname):
    all_lines = []
    with codecs.open(fname, "r", "utf-8") as fin:
	for line in fin:
	    all_lines.append(line.strip())
    return all_lines

fja_lines = read_all_lines(fja)
fen_lines = read_all_lines(fen)
fhf_lines = read_all_lines(fhf)

print len(fja_lines), len(fen_lines), len(fhf_lines)

tot = range(len(fja_lines))
import random
random.shuffle(tot)

fout_en = codecs.open(prefix + ".train.ja-en.en." + ln, "w", "utf-8")
fout_ja = codecs.open(prefix + ".train.ja-en.ja." + ln, "w", "utf-8")
fout_en_hf = codecs.open(prefix + ".train.hf-en.hf." + ln, "w", "utf-8")
fout_en_hf_en = codecs.open(prefix + ".train.hf-en.en." + ln, "w", "utf-8")

ln = int(ln)
for i in tot[:ln]:
    fout_en.write(fen_lines[i] + "\n")
    fout_ja.write(fja_lines[i] + "\n")

for i in tot[ln:]:
    fout_en_hf.write(fhf_lines[i] + "\n")
    fout_en_hf_en.write(fen_lines[i] + "\n")

fout_en.close()
fout_ja.close()
fout_en_hf.close()
fout_en_hf_en.close()

