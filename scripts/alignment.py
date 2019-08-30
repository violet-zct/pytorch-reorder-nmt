# -*- coding: utf-8 -*-

import os
import codecs
import argparse
import time
from collections import defaultdict

import numpy as np


def process(alinged_line, srcWords, tgtWords, reverse=False):
    alignments = set()
    tokens = alinged_line.split('})')
    for i, token in enumerate(tokens):
        if len(token) == 0:
            continue

        tmp = token.strip().split('({')
        srcWord = tmp[0].strip()
        if srcWord == 'NULL' or len(srcWord) == 0:
            continue
        assert srcWord == srcWords[i], 'error: word not match: |' + srcWord.encode('utf-8') + '| |' + srcWords[i].encode('utf-8') + '|'
        

        index = tmp[1].strip().split(' ')
        index = [int(idx) for idx in index if len(idx) > 0]
        for idx in index:
            tgtWord = tgtWords[idx]
            alignment = srcWord + '|||' + tgtWord if not reverse else tgtWord + '|||' + srcWord
            alignments.add(alignment)
    return alignments


    return alignments
def align(forward, backward, output):
    ffop = codecs.open(forward, 'r', 'utf-8')
    bfop = codecs.open(backward, 'r', 'utf-8')
    
    fline = ffop.readline()
    bline = bfop.readline()

    alignments = defaultdict(lambda: 0)
    num_pairs = 1
    while len(fline) > 0:
        assert len(bline) > 0
        # print(fline.strip())
        # print(bline.strip())
        print(num_pairs)

        fline = ffop.readline().strip()
        bline = bfop.readline().strip()

        tokens = fline.split(' ')
        tgtWords = ['NULL'] + [token.strip() for token in tokens] # japanese words

        tokens = bline.split(' ')
        srcWords = ['NULL'] + [token.strip() for token in tokens] # english words

        fline = ffop.readline().strip()
        bline = bfop.readline().strip()

        forward_align = process(fline, srcWords, tgtWords)
        backward_align = process(bline, tgtWords, srcWords, reverse=True)

        for alignment in forward_align.intersection(backward_align):
            # print(alignment.encode('utf-8'))
            alignments[alignment] += 1

        fline = ffop.readline()
        bline = bfop.readline()
        num_pairs += 1

    alignments = sorted(alignments, key=alignments.get, reverse=True)
    with codecs.open(output, 'w', 'utf-8') as ofop:
        for alignment in alignments:
            ofop.write(alignment + '\n')


def main():
    parser = argparse.ArgumentParser(description='word alignment')
    parser.add_argument('--forward', help='path for alignment file of forward direction (ja-en)', required=True)
    parser.add_argument('--backward', help='path for alignment file of backward direction (en-ja)', required=True)
    parser.add_argument('--output', help='path for output file', required=True)

    args = parser.parse_args()

    align(args.forward, args.backward, args.output)


if __name__ == "__main__":
    main()