# -*- coding: utf-8 -*-
import numpy as np
import argparse
from collections import defaultdict
import codecs
import torch
import math
import sys
import io

# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss

    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


FAISS_AVAILABLE = False
def get_batches(n, batch_size):
    tot = math.ceil(n*1.0/batch_size)
    batches = []
    for i in range(tot):
        batches.append((i*batch_size, min((i+1)*batch_size, n)))
    return batches


class vocab():
    def __init__(self, args):
        self.args = args
        self.src_path = args.src_data_path
        self.tgt_path = args.tgt_data_path

        src_word2id = self.build_vocab(self.src_path)
        tgt_word2id = self.build_vocab(self.tgt_path)

        np_src_embeddings, self.src_word2id, self.src_id2word = self.load_embedding_src(args.src_emb_path, src_word2id)
        np_tgt_embeddings, self.tgt_word2id, self.tgt_id2word = self.load_embedding_tgt(args.tgt2src_emb_path, tgt_word2id, np_src_embeddings)

        self.src_emb = torch.from_numpy(np_src_embeddings)
        self.tgt_emb = torch.from_numpy(np_tgt_embeddings)
        
        print(len(self.src_word2id), self.src_emb.size())
        print(len(self.tgt_word2id), self.tgt_emb.size())
        self.dictionary = self.word_translation()

        self.lexicon = self.load_lexicon(args.align_path)

    def build_vocab(self, path, min_freq=5):
        print("Reading data and build initial vocabulary!")
        freqs = defaultdict(lambda: 0)

        with codecs.open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                words = line.strip().split()
                for word in words:
                    freqs[word] += 1
        sorted_words = sorted(freqs, key=freqs.get, reverse=True)

        wordlist = [word for word in sorted_words if freqs[word] > min_freq]

        word2id = dict()
        for i, word in enumerate(wordlist):
            word2id[word] = i
        return word2id

    def load_embedding_src(self, emb_path, word2id):
        new_word2id = dict()

        print("loading embedding from %s" % (emb_path, ))
        embedd_vectors = []
        with codecs.open(emb_path, 'r', 'utf-8') as file:
            # skip the first line
            headline = file.readline().strip()
            voc_size, embedd_dim = headline.split()
            embedd_dim = int(embedd_dim)
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                if vect.shape[0] != 300:
                    print("Dimension error!")
                    continue
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01

                if word in word2id or word.lower() in word2id:
                    word = word.lower() if word.lower() in word2id else word
                    if word in new_word2id:
                        continue
                    else:
                        new_word2id[word] = len(new_word2id)
                        embedd_vectors.append(vect[None, :])
                
        embeddings = np.concatenate(embedd_vectors, 0)
        new_id2word = {v: k for k, v in new_word2id.items()}
        print("Old vocab %d, New vocab found embeddings %d" % (len(word2id), len(new_word2id)))
        return embeddings, new_word2id, new_id2word

    def load_embedding_tgt(self, emb_path, word2id, src_emb):
        new_word2id = dict()
        
        print(src_emb.shape)
        print("loading embedding from %s" % (emb_path,))
        embedd_vectors = []
        with codecs.open(emb_path, 'r', 'utf-8') as file:
            # skip the first line
            headline = file.readline().strip()
            voc_size, embedd_dim = headline.split()
            embedd_dim = int(embedd_dim)
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01

                if vect.shape[0] != 300:
                    print("Dimension error!")
                    continue

                if word in word2id or word.lower() in word2id:
                    word = word.lower() if word.lower() in word2id else word
                    if word in new_word2id:
                        continue
                    else:
                        new_word2id[word] = len(new_word2id)
                        embedd_vectors.append(vect[None, :])

        new_word2id["_va0"] = len(new_word2id)
        # x1, x2 = self.src_word2id[u"は"], self.src_word2id[u"が"]
        x1 = self.src_word2id[u"が"]
        embedd_vectors.append(src_emb[x1][None, :])

        new_word2id["_va1"] = len(new_word2id)
        x, y, z = self.src_word2id[u"は"], self.src_word2id[u"が"], self.src_word2id[u"を"]
        embedding = 1. / 3 * (src_emb[x] + src_emb[y] + src_emb[z])
        embedd_vectors.append(embedding[None, :])

        new_word2id["_va2"] = len(new_word2id)
        x, y, z = self.src_word2id[u"と"], self.src_word2id[u"を"], self.src_word2id[u"に"]
        embedding = 1. / 3 * (src_emb[x] + src_emb[y] + src_emb[z])
        embedd_vectors.append(embedding[None, :])

        embeddings = np.concatenate(embedd_vectors, 0)
        new_id2word = {v: k for k, v in new_word2id.items()}
        print("Old vocab %d, New vocab found embeddings %d" % (len(word2id), len(new_word2id)))
        return embeddings, new_word2id, new_id2word

    def get_nn_avg_dist(self, emb, query, knn):
        """
        Compute the average distance of the `knn` nearest neighbors
        for a given set of embeddings and queries.
        Use Faiss if available.
        """
        if FAISS_AVAILABLE:
            emb = emb.cpu().numpy()
            query = query.cpu().numpy()
            if hasattr(faiss, 'StandardGpuResources'):
                # gpu mode
                res = faiss.StandardGpuResources()
                config = faiss.GpuIndexFlatConfig()
                config.device = 0
                index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
            else:
                # cpu mode
                index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            distances, _ = index.search(query, knn)
            return distances.mean(1)
        else:
            bs = 1024
            all_distances = []
            emb = emb.transpose(0, 1).contiguous()
            for i in range(0, query.shape[0], bs):
                distances = query[i:i + bs].mm(emb)
                best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
                all_distances.append(best_distances.mean(1).cpu())
            all_distances = torch.cat(all_distances)
            return all_distances.numpy()

    def load_lexicon(self, path):
        if self.args.align_number <= 0:
            return None
        # j|||e
        num = 0
        lexion_align = dict()
        with codecs.open(path, "r", "utf-8") as fin:
            for line in fin:
                l = line.strip().lower().split('|||')
                src_word = l[0]
                tgt_word = l[1]
                lexion_align[tgt_word] = src_word
                num += 1

                if num > self.args.align_number:
                    break
        return lexion_align

    def word_translation(self):
        """
        Given source and target word embeddings, and a dictionary,
        evaluate the translation accuracy using the precision@k.

        1 is query, 2 is database
        """
        print("Building dictionary!")
        emb1 = self.tgt_emb
        emb2 = self.src_emb

        # # normalize word embeddings
        emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

        # average distances to k nearest neighbors
        knn = 10

        average_dist1 = self.get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = self.get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        bs = 512
        all_scores = []
        print(average_dist1.size(), average_dist2.size())
        for i, j in get_batches(len(self.tgt_word2id), bs):
            query = emb1[i:j, :]
            scores = query.mm(emb2.transpose(0, 1))
            scores.mul_(2)
            scores.sub_(average_dist1[i:j][:, None])
            scores.sub_(average_dist2[None, :])
            all_scores.append(scores)

        all_scores = torch.cat(all_scores, 0)
        top_matches = all_scores.topk(10, 1, True)[1]

        dictionary = list(top_matches[:, 0].cpu().numpy())

        identical = 0
        if self.args.copy_id:
            for i, j in enumerate(dictionary):
                if self.tgt_id2word[i] in self.src_word2id:
                    dictionary[i] = self.src_word2id[self.tgt_id2word[i]]
                    identical += 1
        print("In total there are %d identical words indexed in the dictionary!" % identical)
        for i in range(0, 100):
            print(self.tgt_id2word[i], self.src_id2word[dictionary[i]])

        ii = self.tgt_word2id["_va0"]
        print("_va0", self.src_id2word[dictionary[ii]])
        jj = self.tgt_word2id["_va1"]
        print("_va1", self.src_id2word[dictionary[jj]])
        kk = self.tgt_word2id["_va2"]
        print("_va2", self.src_id2word[dictionary[kk]])

        return dictionary


def main(args):
    database = vocab(args)
    avg_trans_ratios = 0.
    tot_sents = 0
    fout = io.open(args.output_path, "w", encoding="utf-8")
    with io.open(args.tgt_path_for_tran, "r", encoding="utf-8") as fin:
        for line in fin:
            tokens = line.strip().split()
            translation = []
            translated = 0
            for t in tokens:
                if database.lexicon is not None and t in database.lexicon:
                    translation.append(database.lexicon[t])
                    translated += 1
                elif t not in database.tgt_word2id:
                    translation.append(t)
                else:
                    tran = database.src_id2word[database.dictionary[database.tgt_word2id[t]]]
                    translation.append(tran)
                    translated += 1
            avg_trans_ratios += (translated * 1.) / len(translation)
            tot_sents += 1
            fout.write(" ".join(translation) + "\n")
    print("Average translation ratios = %f" % (avg_trans_ratios / tot_sents))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_path", type=str)
    parser.add_argument("--tgt_data_path", type=str)
    parser.add_argument("--src_emb_path", type=str)
    parser.add_argument("--tgt2src_emb_path", type=str)
    parser.add_argument("--tgt_path_for_tran", type=str)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--align_path", type=str, default=None)
    parser.add_argument("--align_number", type=int)
    parser.add_argument("--copy_id", type=int, default=0, help="if not translate identical words")
    args = parser.parse_args()
    main(args)
