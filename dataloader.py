import numpy as np
from collections import defaultdict
import codecs
import os
import cPickle as pkl

def get_sorted_wordlist(paths, min_freq=1):
    freqs = defaultdict(lambda: 0)
    for path in paths:
        if path is None:
            continue
        with codecs.open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                words = line.strip().split()
                for word in words:
                    freqs[word] += 1
    sorted_words = sorted(freqs, key=freqs.get, reverse=True)

    wordlist = [word for word in sorted_words if freqs[word] > min_freq]
    return wordlist

class NMT_Dataloader():
    def __init__(self, args):
        self.args = args
        self.sup_train_src_path = args.sup_train_src_path
        self.sup_train_tgt_path = args.sup_train_tgt_path
        self.unsup_train_src_path = args.unsup_train_src_path

        if False and os.path.exists(args.src_vocab_pkl_path) and os.path.exists(args.tgt_vocab_pkl_path):
            with open(args.src_vocab_pkl_path, "rb") as src, open(args.tgt_vocab_pkl_path, "rb") as tgt:
                self.src_vocab = pkl.load(src)
                self.src_vocab_info = pkl.load(src)
                self.tgt_vocab = pkl.load(tgt)
                self.tgt_vocab_info = pkl.load(tgt)
        else:
            src_paths = [self.sup_train_src_path]
            tgt_paths = [self.sup_train_tgt_path]
            self.tgt_vocab, self.tgt_vocab_info, self.src_vocab, self.src_vocab_info = self.get_vocab(src_paths,
                                                                                                      tgt_paths,
                                                                                                      min_freq=1,
                                                                                                      src_voc_size=args.src_vocab_size,
                                                                                                      tgt_voc_size=args.tgt_vocab_size,
                                                                                                      src_saveto=args.src_vocab_pkl_path,
                                                                                                      tgt_saveto=args.tgt_vocab_pkl_path)

        self.src_voc_size = len(self.src_vocab)
        self.tgt_voc_size = len(self.tgt_vocab)

        self.src_id_to_word = {v: k for k, v in self.src_vocab.iteritems()}
        self.tgt_id_to_word = {v: k for k, v in self.tgt_vocab.iteritems()}

        # args.max_sent_len = self.set_max_len()
        # print("Max sent len: %d" % args.max_sent_len)
        print("Source and target vocab size: %d, %d" % (self.src_voc_size, self.tgt_voc_size))

    def get_vocab(self, src_paths, tgt_paths, src_voc_size=30000, tgt_voc_size=30000, min_freq=1, src_saveto=None, tgt_saveto=None):
        src_words = get_sorted_wordlist(src_paths, min_freq=min_freq)
        tgt_words = get_sorted_wordlist(tgt_paths, min_freq=min_freq)

        src_vocab = {'<es>': 0, '<bs>': 1, '<unk>': 2, "<pad>": 3}
        tgt_vocab = {'<es>': 0, '<bs>': 1, '<unk>': 2, "<pad>": 3}

        for tgt_word in tgt_words:
            if len(tgt_vocab) > tgt_voc_size:
                break
            tgt_vocab[tgt_word] = len(tgt_vocab)

        tgt_vocab_info = {"vocab_size": len(tgt_vocab),
                            "eos": tgt_vocab['<es>'], "bos": tgt_vocab['<bs>'],
                            "unk": tgt_vocab['<unk>'], "pad": tgt_vocab['<pad>']}

        for src_word in src_words:
            if len(src_vocab) > src_voc_size:
                break
            src_vocab[src_word] = len(src_vocab)

        src_vocab_info = {"vocab_size": len(src_vocab),
                               "eos": src_vocab['<es>'], "bos": src_vocab['<bs>'],
                               "unk": src_vocab['<unk>'], "pad": src_vocab['<pad>']}

        if src_saveto is not None:
            with open(src_saveto, "wb") as fin:
                pkl.dump(src_vocab, fin)
                pkl.dump(src_vocab_info, fin)

        if tgt_saveto is not None:
            with open(tgt_saveto, "wb") as fin:
                pkl.dump(tgt_vocab, fin)
                pkl.dump(tgt_vocab_info, fin)

        return tgt_vocab, tgt_vocab_info, src_vocab, src_vocab_info

    def read_file(self, paths, vocab, pad=True):
        dataset = []
        for path in paths:
            with codecs.open(path,  "r", "utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if pad:
                        sent = [vocab["<bs>"]] + [vocab[word] if word in vocab else vocab["<unk>"] for word in line.split()] + \
                            [vocab["<es>"]]
                    else:
                        sent = [vocab[word] if word in vocab else vocab["<unk>"] for word in line.split()]
                    dataset.append(sent)
        return dataset

    def read_dataset(self):
        # for train
        sup_train_src = self.read_file([self.sup_train_src_path], self.src_vocab, pad=self.args.pad_src_sent)
        sup_train_tgt = self.read_file([self.sup_train_tgt_path], self.tgt_vocab)

        return sup_train_src, sup_train_tgt

    def read_dataset_from_path(self, src_path, tgt_path):
        src_set = self.read_file([src_path], self.src_vocab, pad=self.args.pad_src_sent)
        tgt_set = self.read_file([tgt_path], self.tgt_vocab)
        return src_set, tgt_set
