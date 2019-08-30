# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import codecs

hf = sys.argv[1]
sn = sys.argv[2]

lsn = []
with open(sn, "r") as fin:
    for line in fin:
	lsn.append(int(line.strip()))

i = 0
fout = codecs.open("recovered.hf", "w", "utf-8")
with codecs.open(hf, "r", "utf-8") as fin:
    cc = 0
    ss = ""
    for line in fin:
	if cc == lsn[i]:
	    fout.write(ss.strip() + "\n")
	    cc = 1
	    ss = line.strip()
	    i += 1
	else:
	    ss = ss + " " + line.strip()
	    cc += 1
fout.write(ss.strip() + "\n")
i += 1
fout.close()
print len(lsn), i
