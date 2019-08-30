# -*- coding: utf-8 -*-
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import codecs

def read_all_lines(fname):
    all_lines = []
    with codecs.open(fname, "r", "utf-8") as fin:
	for line in fin:
	    all_lines.append(line.strip())
    return all_lines

fja = "kyoto-train.lower.ja"
fen = "kyoto-train.lower.en"

fsplit = "./tokenizer/full_sents.split"

fja_lines = read_all_lines(fja)
fen_lines = read_all_lines(fen)
fsplit_lines = read_all_lines(fsplit)

print len(fsplit_lines)
print len(fja_lines), len(fen_lines)

fnew_en = codecs.open("kyoto-train-final.lower.en", "w", "utf-8")
fnew_ja = codecs.open("kyoto-train-final.lower.ja", "w", "utf-8")

j = 0
i = 0

while True:
    while fsplit_lines[i] != "<P>":
        if fsplit_lines[i] not in fen_lines[j]:
            print fsplit_lines[i]
            print fen_lines[j]
            j += 1
        else:
            i += 1
    fnew_en.write(fen_lines[j] + "\n")
    fnew_ja.write(fja_lines[j] + "\n")
    j += 1
    i += 1
    if i >= len(fsplit_lines):
        break
print j