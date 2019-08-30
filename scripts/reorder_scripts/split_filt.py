import os
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

inp_name = sys.argv[1]
split_num = int(sys.argv[2])

dir_name = "split_files"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
import codecs
import numpy as np

counter = 0
lines = []
tot_input_sents = 0
sent_nums = []

f_blank_line = codecs.open("blank_line.en", "w", "utf-8")

with codecs.open(inp_name, "r", "utf-8") as fin:
    for line in fin:
        f_blank_line.write(line.strip() + "\n\n")
        tot_input_sents += 1

f_blank_line.close()
os.system("perl split-sentences.perl -l en < blank_line.en > full_sents.split")

with codecs.open("full_sents.split", "r", "utf-8") as fin:
    nn = 0
    for line in fin:
        if line.strip() == "<P>":
            assert nn >= 1
            sent_nums.append(nn)
            nn = 0
        else:
            lines.append(line.strip())
            counter += 1
            nn += 1

print len(sent_nums), tot_input_sents
print counter, sum(sent_nums)

if split_num == -1:
    exit()

with open("sent.split.num", "w") as fin:
    for n in sent_nums:
        fin.write(str(n) + "\n")

batch_size = counter // split_num
print "batch_size", batch_size
tot_write = 0
for i in range(split_num):
    fout = codecs.open(os.path.join(dir_name, str(i) + ".out"), "w", "utf-8")
    for j in range(batch_size):
        fout.write(lines[i * batch_size + j].strip() + "\n")
        tot_write += 1
    fout.close()

fout = codecs.open(os.path.join(dir_name, str(split_num) + ".out"), "w", "utf-8")
if batch_size * split_num < counter:
    for i in range(split_num * batch_size, len(lines)):
        fout.write(lines[i].strip() + "\n")
        tot_write += 1
    fout.close()
print tot_write, len(lines)
