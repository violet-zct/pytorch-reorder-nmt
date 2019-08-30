import sys
import os

tot = int(sys.argv[1])

flist = []
ss = ""
for i in range(tot):
    flist.append(str(i)+".xml.parse")
    ss += "%s "
args = tuple(flist) + ("full.xml.parse", )
cmd = "cat " + ss + "> %s"
print args
print cmd
os.system(cmd % args)

