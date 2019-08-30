# -*- coding: utf-8 -*-
import codecs
import uuid

import math
import logging
from collections import defaultdict
import operator

import codecs
import uuid
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=750)
import itertools
import copy
from random import shuffle
import time
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def get_batches(data_length, batch_size):
    num_batch = int(np.ceil(data_length/float(batch_size)))
    return [(i*batch_size, min(data_length, (i+1)*batch_size)) for i in range(0, num_batch)]


def get_total_words(seqs):
    # return len(seqs)
    total_words = 0
    for s in seqs:
        total_words += len(s) - 1
    return total_words


def transpose_input_var(seq, padding_token=3, is_test=False, is_cuda=True):
    # return Variable(max_len, batch_size)
    max_len = max([len(sent) for sent in seq])
    seq_pad = []
    seq_mask = []
    for i in range(max_len):
        pad_temp = [sent[i] if i < len(sent) else padding_token for sent in seq]
        mask_temp = [1 if i < len(sent) else 0 for sent in seq]
        seq_pad.append(pad_temp)
        seq_mask.append(mask_temp)

    seq_pad = Variable(torch.LongTensor(seq_pad), volatile=is_test, requires_grad=False)
    seq_mask = Variable(torch.FloatTensor(seq_mask), volatile=is_test, requires_grad=False)
    if is_cuda:
        seq_pad = seq_pad.cuda()
        seq_mask = seq_mask.cuda()
    return seq_pad, seq_mask


def pad_input_var(seq, padding_token=3, is_test=False, is_cuda=True):
    # not the transposed version
    # return Variable(batch_size, max_len)
    batch_size = len(seq)
    max_len = max([len(sent) for sent in seq])
    seq_pad = []
    seq_mask = []
    seq_len = [len(sent) for sent in seq]

    for s in seq:
        temp_pad = s if len(s) == max_len else s + [padding_token] * (max_len - len(s))
        temp_mask = [1] * len(s) if len(s) == max_len else [1] * len(s) + [0] * (max_len - len(s))
        seq_pad.append(temp_pad)
        seq_mask.append(temp_mask)

    seq_pad = Variable(torch.LongTensor(seq_pad), volatile=is_test, requires_grad=False)
    seq_mask = Variable(torch.FloatTensor(seq_mask), volatile=is_test, requires_grad=False)

    if is_cuda:
        seq_pad = seq_pad.cuda()
        seq_mask = seq_mask.cuda()
    return seq_pad, seq_mask, seq_len


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def get_sent_idx(sent, vocab, s=1, e=0, unk=2, pad=True):
    sent = [vocab[w] if w in vocab else unk for w in sent]
    if pad:
        sent = [s] + sent + [e]
    return sent


def make_bucket_batches(data_pair, batch_size):
    buckets = defaultdict(list)
    for pair in data_pair:
        src = pair[0]
        buckets[len(src)].append(pair)

    batches = []
    src_lens = buckets.keys()
    shuffle(src_lens)
    for src_len in src_lens:
        bucket = buckets[src_len]
        np.random.shuffle(bucket)
        num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
            s = [bucket[i * batch_size + j][0] for j in range(cur_batch_size)]
            t = [bucket[i * batch_size + j][1] for j in range(cur_batch_size)]
            b_ids = sorted(range(cur_batch_size), key=lambda x: len(t[x]), reverse=True)
            t = [t[i] for i in b_ids]
            batches.append((s, t))

    np.random.shuffle(batches)
    return batches


def make_bucket_single_batches(data, batch_size):
    buckets = defaultdict(list)
    for d in data:
        buckets[len(d)].append(d)

    batches = []
    src_lens = buckets.keys()
    shuffle(src_lens)
    for src_len in src_lens:
        bucket = buckets[src_len]
        np.random.shuffle(bucket)
        num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
            batches.append([bucket[i * batch_size + j] for j in range(cur_batch_size)])

    np.random.shuffle(batches)
    return batches


def data_iterator(data_pair, batch_size, sorted=False):
    if sorted:
        batches = make_sorted_pair_batches(data_pair, batch_size)
    else:
        batches = make_bucket_batches(data_pair, batch_size)
    for batch in batches:
        yield batch


def sort_pairs_and_make_batches(a, b, batch_size):
    a_ids = sorted(range(len(a)), key=lambda x: len(a[x]), reverse=True)
    b_ids = sorted(range(len(b)), key=lambda x: len(b[x]), reverse=True)

    return make_bucket_batches(zip([a[i] for i in a_ids], [b[i] for i in b_ids]), batch_size)


def make_sorted_pair_batches(data_pair, batch_size):
    buckets = [20, 30, 40, 50, 60, 80]
    grouped_data = [[] for _ in buckets]

    outlier = []
    for pair in data_pair:
        src_len = len(pair[0])
        if src_len > buckets[-1]:
            outlier.append(pair)
            continue

        for bid, i in enumerate(buckets):
            if src_len <= i:
                grouped_data[bid].append(pair)
                break

    if False and len(outlier) > 0:
        grouped_data.append(outlier)

    batches = []
    # shuffle(grouped_data)
    for group in grouped_data:
        shuffle(group)
        num_batches = int(np.ceil(len(group) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(group) - batch_size * i
            src_batch = [group[i * batch_size + j][0] for j in range(cur_batch_size)]
            tgt_batch = [group[i * batch_size + j][1] for j in range(cur_batch_size)]
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_batch[src_id]), reverse=True)
            batches.append(([src_batch[kk] for kk in src_ids], [tgt_batch[kk] for kk in src_ids]))
    shuffle(batches)
    return batches


def make_sorted_single_batches(data, batch_size):
    buckets = [20, 30, 40, 50, 60, 80, 100, 120]
    grouped_data = [[] for _ in buckets]

    for d in data:
        src_len = len(d)
        for bid, i in enumerate(buckets):
            if src_len <= buckets[i]:
                grouped_data[bid].append(d)
                break
    batches = []
    shuffle(grouped_data)
    for group in grouped_data:
        shuffle(group)
        num_batches = int(np.ceil(len(group) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(group) - batch_size * i
            src_batch = [group[i * batch_size + j][0] for j in range(cur_batch_size)]
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_batch[src_id]), reverse=True)
            batches.append([src_batch[kk] for kk in src_ids])
    shuffle(batches)
    return batches


def data_double_pair_single_iterator(data_pair_1, data_2, batch_size, sorted=False):
    # make batches for (s1, t1), (s)
    if sorted:
        batches_1 = make_sorted_pair_batches(data_pair_1, batch_size)
        batches_2 = make_sorted_single_batches(data_2, batch_size)
    else:
        batches_1 = make_bucket_batches(data_pair_1, batch_size)
        batches_2 = make_bucket_single_batches(data_2, batch_size)

    len_1 = len(batches_1)
    len_2 = len(batches_2)
    if len_1 > len_2:
        batches_2.extend(batches_2[:len_1-len_2])
    else:
        batches_1.extend(batches_1[:len_2-len_1])

    for batch in zip(batches_1, batches_2):
        yield batch


def make_batches(batch_size, *data_list):
    # when use: make_batches(4, a, b)
    num_dt_src = len(data_list)
    data_pairs = zip(*data_list)
    buckets = defaultdict(list)
    for d in data_pairs:
        buckets[len(d[0])].append(d)

    batches = []
    src_lens = buckets.keys()
    shuffle(src_lens)
    for src_len in src_lens:
        bucket = buckets[src_len]
        np.random.shuffle(bucket)
        num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
            batch_sources = ()
            for k in range(num_dt_src):
                data_item = []
                for j in range(cur_batch_size):
                    data_item.append(bucket[i * batch_size + j][k])
                batch_sources += (data_item, )
            batches.append(batch_sources)
    np.random.shuffle(batches)
    return batches


def data_multi_source_iterator(batch_size, *data_list):
    '''
    :param batch_size:
    :param data_list: all items in data_list have the same number of examples
    all the data will be bucketed based on the length of the first item in data_list
    :return:
    '''
    batches = make_batches(batch_size, *data_list)
    for batch in batches:
        yield batch


def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents


def data_iter_sort(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle: np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def cal_bleu(fref, translations, foutput, bpe=None):
    uid = uuid.uuid4().get_hex()[:6]
    empty = True
    with codecs.open(foutput, "w", "utf-8") as fout:
        for line in translations:
            if len(line) > 0 and not all(x == u"" for x in line):
                empty = False
            if bpe == "spm":
                line = ''.join(line)
                line = line.replace('▁', ' ').strip()
            else:
                line = " ".join(line)
            fout.write(line + "\n")
    if empty:
        os.system("rm %s" % (foutput))
        return 0.0

    if bpe == "subword":
        frestore = foutput + ".restore"
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (foutput, frestore))
        os.system("mv %s %s" % (frestore, foutput))

    os.system("perl ../Datasets/multi-bleu.perl %s < %s > %s_bleu_score_temp" % (fref, foutput, str(uid)))
    output = open(str(uid) + "_bleu_score_temp", "r").read().strip()
    bleu = float(output.split(",")[0].split("=")[1].strip())

    os.system("rm %s_bleu_score_temp" % (str(uid)))
    return bleu


def dev_ppl(args, dataloader, model):
    dev_src, dev_tgt = dataloader.read_dataset_from_path(args.valid_src_path, args.valid_tgt_path)

    cum_ppl = cum_words = 0
    model.eval()
    for batch in data_iterator(zip(dev_src, dev_tgt), args.batch_size, sorted=args.sort_src):
        src_batch, tgt_batch = batch[0], batch[1]
        loss = model.forward(src_batch, tgt_batch)
        if len(loss) > 1:
            loss = loss[0]
        tgt_word_num = sum([len(s)-1 for s in tgt_batch])

        loss_val = loss.data[0]
        cum_ppl += loss_val
        cum_words += tgt_word_num

    ppl = np.exp(cum_ppl / cum_words)
    return ppl


def translate(model, foutput, src_path, tgt_path, src_word_to_id, tgt_id_to_word, bpe=None, pad=True, reorder_inp=False):
    translations = []
    references = []
    tot_dev_sent = 0

    src_sents = []
    tgt_sents = []
    src_lines = []
    tgt_word_to_id = {v: k for k, v in tgt_id_to_word.iteritems()}

    cum_ppl = tot_words = 0.0
    with codecs.open(src_path, "r", "utf-8") as fsrc, codecs.open(tgt_path, "r", "utf-8") as ftgt:
        for src_line, tgt_line in zip(fsrc, ftgt):
            src_sent = get_sent_idx(src_line.strip().split(), src_word_to_id,  pad=pad)

            # Beam-search

            scores, samples = model.evaluate(src_sent, 100)

            if type(samples[0]) is not list:
                sample = samples
                ll = scores
            else:
                scores = scores / np.array([len(s) for s in samples])
                sample = samples[np.array(scores).argmax()]
                ll = np.array(scores).max()

            hyp = [tgt_id_to_word[w] for w in sample]
            translations.append(hyp)

            src_lines.append(src_line)
            src_sents.append(src_sent)
            tgt_sents.append(get_sent_idx(tgt_line.strip().split(), tgt_word_to_id))
            references.append(tgt_line)

            hyp = u" ".join(hyp)
            cum_ppl += ll
            tot_words += len(hyp) + 2
            print("*"*100)
            print("Src sent: %s" % src_line.strip())
            print("Tgt sent: %s" % tgt_line.strip())
            print("Hypothesis: %s " % hyp)

            tot_dev_sent += 1

        bleu_score = cal_bleu(tgt_path, translations, foutput, bpe)
    return bleu_score


def load_embedding_dict(embedding, embedding_path):
    print("loading embedding: %s from %s" % (embedding, embedding_path))
    if embedding == 'fasttext':
        embedd_dict = dict()
        with codecs.open(embedding_path, 'r', 'utf-8') as file:
            # skip the first line
            headline = file.readline().strip()
            voc_size, embedd_dim = headline.split()
            voc_size = int(voc_size)
            embedd_dim = int(embedd_dim)

            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if len(tokens) < embedd_dim:
                    print(tokens)
                    continue

                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                start = len(tokens) - embedd_dim
                word = ' '.join(tokens[0:start])
                embedd[:] = tokens[start:]
                if word in embedd_dict:
                    print("Duplicate word!")
                    continue
                embedd_dict[word] = embedd

        # assert len(embedd_dict) == voc_size, "error of loading embedding: %d not equal to %d" % (len(embedd_dict), voc_size)
        print("%d, %d" % (voc_size, len(embedd_dict)))
        return embedd_dict, embedd_dim
    else:
        raise ValueError("embedding should choose from [fasttext, ]")
