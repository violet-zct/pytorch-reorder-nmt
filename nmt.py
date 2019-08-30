import sys
reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append("..")
from utils import *
from nn_modules import *

from model import Model
import argparse
import dataloader as DataLoader
from args import *
import random

parser = argparse.ArgumentParser()
add_NMT_model_args(parser)


class NMT(Model):
    def __init__(self, args, src_vocab_info, tgt_vocab_info):
        super(NMT, self).__init__(args)

        self.hidden_dim = args.hidden_dim
        self.emb_dim = args.emb_dim

        self.src_vocab_info = src_vocab_info
        self.tgt_vocab_infor = tgt_vocab_info
        # self.dec_ctx_dim = self.hidden_dim
        self.dropout_rate = args.normal_dropout

        att_dim = args.dec_att_dim
        self.encoder = BiLSTM_Encoder(args, self.emb_dim, self.hidden_dim, src_vocab_info)
        self.dec_ctx_dim = self.encoder.output_dim
        # ctx feed works better than readout feed
        self.decoder = NMT_decoder(args, self.hidden_dim, self.emb_dim, self.dec_ctx_dim, tgt_vocab_info, init_dim=self.dec_ctx_dim,
                                   att_dim=att_dim, att_type=args.att_type, cov_att=args.cov_att)

    def forward(self, src_batch, tgt_batch):
        '''
        :param src_batch: list of lists - unpadded word tokens
        :param tgt_batch: list of lists - unpadded word tokens
        :return:
        '''
        transpose_tgt_seqs, transpose_tgt_masks = transpose_input_var(tgt_batch, is_test=not self.training, is_cuda=self.args.cuda)

        if self.args.init_dec == "cell":
            src_encodings, src_masks, dec_init_cell = self.encoder.forward(src_batch)
            dec_init_hidden = None
        else:
            src_encodings, src_masks, dec_init_cell, dec_init_hidden = self.encoder.forward(src_batch)
        loss = self.decoder.forward(src_encodings, transpose_tgt_seqs, src_masks, transpose_tgt_masks, dec_init_cell, dec_init_hidden)

        return loss

    def evaluate(self, src_sent, max_len=100):
        if self.args.init_dec == "cell":
            src_encodings, src_masks, dec_init_cell = self.encoder.forward([src_sent])
            dec_init_hidden = None
        else:
            src_encodings, src_masks, dec_init_cell, dec_init_hidden = self.encoder.forward([src_sent])

        scores, translations = self.decoder.beam_search_elem(src_encodings, dec_init_cell, max_len, dec_init_hidden=dec_init_hidden)
        return scores, translations


def main(args):
    print("Loading data....")
    dataloader = DataLoader.NMT_Dataloader(args)

    sup_train_src, sup_train_tgt = dataloader.read_dataset()
    src_vocab_info, tgt_vocab_info = dataloader.src_vocab_info, dataloader.tgt_vocab_info
    epoches = args.epoch
    epoch = 1
    valid_freq = args.valid_freq
    display_freq = args.display_freq
    lr = args.init_lr

    bad_counter = updates = cum_loss = cum_ppl = report_loss = report_ppl = cum_example = cum_words = report_example = report_words = 0
    patience = 5
    decay_number = best_metric = best_ppl = 0

    valid_history = []
    begin_training_time = time.time()

    model = NMT(args, src_vocab_info, tgt_vocab_info)
    if args.cuda:
        model.cuda()
    if args.uniform_init > 0.0:
        print("Uniformly initialize parameters.")
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    while epoch < epoches:
        for labeled_batch in data_iterator(zip(sup_train_src, sup_train_tgt), args.batch_size, sorted=args.sort_src):
            l_src_batch, l_tgt_batch = labeled_batch
            # u_src_batch = unlabeled_batch
            updates += 1

            model.train()
            opt.zero_grad()

            loss = model.forward(l_src_batch, l_tgt_batch)
            tgt_word_num = sum([len(s)-1 for s in l_tgt_batch])
            batch_size = len(l_src_batch)

            # word_loss = loss / tgt_word_num
            # word_loss.backward()
            avg_loss = loss / batch_size
            avg_loss.backward()
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            opt.step()

            loss_val = loss.data[0]
            cum_loss += loss_val
            cum_ppl += loss_val
            cum_example += batch_size
            cum_words += tgt_word_num
            report_loss += loss_val
            report_ppl += loss_val
            report_example += batch_size
            report_words += tgt_word_num

            if updates % display_freq == 0:
                print("Epoch=%d, Step=%d, cum loss=%.3f, cum ppl=%.3f, loss/sent=%.3f, Speed=%.2f words/sec" %
                      (epoch, updates, report_loss/report_example, np.exp(report_ppl/report_words), loss_val/batch_size, report_words/(time.time()-begin_training_time)))
                report_loss = report_ppl = report_words = report_example = 0.
                begin_training_time = time.time()

            if updates % valid_freq == 0:
                print("Epoch=%d, Step=%d, cum loss=%.3f, cum ppl=%.3f" %
                      (epoch, updates, cum_loss / cum_example, np.exp(cum_ppl / cum_words)))
                cum_loss = cum_ppl = cum_example = cum_words = 0.

                print("*" * 25 + " Evaluating BLEU on valid set " + "*" * 25)
                begin_time = time.time()
                model.eval()
                if args.valid_metric == "bleu":
                    metric = translate(model, args.translation_output, args.valid_src_path, args.valid_tgt_path,
                                     dataloader.src_vocab, dataloader.tgt_id_to_word, args.bpe, pad=args.pad_src_sent)
                    ss = "BLEU"
                else:
                    metric = dev_ppl(args, dataloader, model)
                    ss = "Dev ppl"
                print("%s score=%f, Evaluation on valid set took %f sec. " % (ss, metric, time.time() - begin_time))

                if len(valid_history) == 0 or (metric > max(valid_history) and (ss == "BLEU" or ss == "Accuracy")) or (metric < min(valid_history) and ss == "Dev ppl"):
                    bad_counter = 0
                    best_metric = metric
                    print("Saving the best model so far....")
                    model.save()
                else:
                    bad_counter += 1
                    print("Hit patience %d." % (bad_counter,))
                    if bad_counter >= 3:
                        model.load()
                        decay_number += 1
                        lr *= 0.5
                        print("Epoch = %d, Learning Rate is Decayed to = %f." % (epoch, lr))
                        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999), eps=1e-8)
                        bad_counter = 0
                if decay_number > patience:
                    # Evaluate on Test data
                    model.load()

                    if args.valid_metric == "ppl":
                        valid_bleu = translate(model, args.translation_output, args.valid_src_path, args.valid_tgt_path,
                                               dataloader.src_vocab, dataloader.tgt_id_to_word, args.bpe, pad=args.pad_src_sent)
                    bleu = translate(model, args.translation_output, args.test_src_path, args.test_tgt_path,
                                     dataloader.src_vocab,
                                     dataloader.tgt_id_to_word, args.bpe, pad=args.pad_src_sent)
                    print("Best %s on valid set: %.3f" % (ss, best_metric))
                    if args.valid_metric == "ppl":
                        print("Valid BLEU score=%f, Evaluation on valid set took %f sec. " % (
                            valid_bleu, time.time() - begin_time))
                    print("BLEU on test set: %.3f" % bleu)
                    print("Early stop!")
                    exit(0)
                valid_history.append(metric)

        epoch += 1

    # Evaluate on Test data
    model.load()
    model.eval()
    bleu = translate(model, args.translation_output, args.test_src_path, args.test_tgt_path, dataloader.src_vocab,
                         dataloader.tgt_id_to_word, args.bpe, pad=args.pad_src_sent)
    print("BLEU on test set: %.3f" % bleu)


def test(args):
    print("Loading data....")
    dataloader = DataLoader.NMT_Dataloader(args)
    src_vocab_info, tgt_vocab_info = dataloader.src_vocab_info, dataloader.tgt_vocab_info
    model = NMT(args, src_vocab_info, tgt_vocab_info)
    if args.cuda:
        model.cuda()
    # Evaluate on Test data
    model.load()
    model.eval()
    metric_value = translate(model, args.translation_output, args.test_src_path, args.test_tgt_path, dataloader.src_vocab,
                             dataloader.tgt_id_to_word, args.bpe, pad=args.pad_src_sent)
    print("BLEU on test set: %.3f" % metric_value)


if __name__ == "__main__":
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    args.save_to = args.save_to + args.model_name + ".model"
    args.translation_output = args.model_name + ".tran"
    if args.load_from is None:
        args.load_from = args.save_to

    print(args)
    if args.test:
        test(args)
    else:
        main(args)