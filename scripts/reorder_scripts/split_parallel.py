import sys
import codecs
import random

random.seed(199203208)

label = sys.argv[1]

# lower and tokenized
en_inp = "train.low.en"
tr_inp = "train.low.tr"

def read_file(path):
    data = []
    with codecs.open(path, "r", "utf-8") as fin:
        for line in fin:
            data.append(line.strip())
    return data

en_data = read_file(en_inp)
tr_data = read_file(tr_inp)

assert len(en_data) == len(tr_data)

inds = range(len(en_data))
random.shuffle(inds)

en_labeled_opt = codecs.open("wmt_sup_train_" + str(label) + ".en", "w", "utf-8")
tr_labeled_opt = codecs.open("wmt_sup_train_" + str(label) + ".tr", "w", "utf-8")
en_ae_opt = codecs.open("wmt_ae_train_" + str(label) + ".en", "w", "utf-8")
tr_ae_opt = codecs.open("wmt_ae_train_" + str(label) + ".tr", "w", "utf-8")

i = 0
label = int(label)
for l_en, l_tr in zip(en_data, tr_data):
    if i % 1000 == 0:
        print "Processed %d lines!" % i
    if i < label:
        en_labeled_opt.write(l_en + "\n")
        tr_labeled_opt.write(l_tr + "\n")
    else:
        en_ae_opt.write(l_en + "\n")
        tr_ae_opt.write(l_tr + "\n")
    i += 1

en_labeled_opt.close()
tr_labeled_opt.close()
en_ae_opt.close()
tr_ae_opt.close()