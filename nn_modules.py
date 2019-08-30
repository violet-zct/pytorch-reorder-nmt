from utils import *
from torch.nn.parameter import Parameter
import torch.nn.init as init

class BiLSTM_Encoder(nn.Module):
    def __init__(self, args, input_size, hidden_size, vocab_info, batch_first=True):
        super(BiLSTM_Encoder, self).__init__()

        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size // 2 if args.half_enc_h_dim else hidden_size
        self.num_layers = args.enc_layers

        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                 batch_first=batch_first, bidirectional=True, dropout=args.normal_dropout)

        self.vocab_emb = nn.Embedding(vocab_info["vocab_size"], input_size, padding_idx=vocab_info["pad"])
        self.output_dim = self.hidden_size * 2
        if args.drop_emb > 0.0:
            self.drop_emb = nn.Dropout(args.drop_emb)

    def forward(self, sents):
        '''
        :param sents: list of unpadded sentences
        :return:
        '''
        # var: (batch_size, max_len) TODO: return seqs sorted by lengths; and src_sents_len
        padded_sent, sent_masks, sent_lens = pad_input_var(sents, is_test=not self.training, is_cuda=self.args.cuda)
        # var: (batch_size, max_len, emb_dim)
        word_embs = self.vocab_emb(padded_sent)
        if self.args.drop_emb > 0.0:
            word_embs = self.drop_emb(word_embs)

        packed_src_embed = pack_padded_sequence(word_embs, sent_lens, batch_first=True)
        # outputs: (batch_size, max_len, hidden_dim*2); last_cell: (2, batch_size, hidden_dim)
        outputs, (last_hidden, last_cell) = self.bilstm.forward(packed_src_embed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # (batch_size, hidden_dim*2)
        dec_init_hidden = torch.cat([last_hidden[-2], last_hidden[-1]], 1)
        dec_init_cell = torch.cat([last_cell[-2], last_cell[-1]], 1)

        if self.args.init_dec == "cell":
            return outputs, sent_masks, dec_init_cell
        else:
            return outputs, sent_masks, dec_init_cell, dec_init_hidden

    def get_vocab_emb(self):
        return self.vocab_emb


class decoder_common(nn.Module):
    def __init__(self, args, init_dim, hidden_dim, emb_dim, ctx_dim, att_dim=-1, att_type="D", cov_att=False):
        super(decoder_common, self).__init__()

        self.args = args
        self.cuda = args.cuda
        self.hidden_dim = hidden_dim
        self.init_dim = init_dim
        self.emb_dim = emb_dim
        self.ctx_dim = ctx_dim

        if args.input_feed == "ctx":
            self.input_size = ctx_dim + emb_dim # input_feeding, concat the last context vector and last prediction embedding
            self.input_feed_dim = ctx_dim
        else:
            self.input_size = hidden_dim + emb_dim # concat last readout and last embedding
            self.input_feed_dim = hidden_dim
        self.dec_rnn = nn.LSTMCell(self.input_size, hidden_dim)

        self.init_dec = init_dim > 0
        if self.init_dec:
            self.dec_initializer = nn.Linear(init_dim, hidden_dim)
            if args.init_dec == "respect":
                self.dec_initializer_h = nn.Linear(init_dim, hidden_dim) #!!!!!!!!!!!!

        self.att_type = att_type

        if att_type == "M":
            # use MLP attention
            assert att_dim > 0
            self.att_hid = nn.Linear(hidden_dim, att_dim, bias=False)
            self.att_V = nn.Linear(att_dim, 1, bias=False)
        else:
            # use Dot product attention
            att_dim = hidden_dim

        # attention: project src features into hidden space
        self.att_src = nn.Linear(ctx_dim, att_dim, bias=False)
        self.cov_att = cov_att
        if cov_att:
            self.trainable_lambd = Parameter(torch.zeros(1), requires_grad=True)
        self.dropout = nn.Dropout(args.normal_dropout)

    def attention(self, h_t, src_transform_encoding, src_encodings, mask=None, prev_att_sum=None):
        '''
        :param h_t: (batch_size, hidden_dim)
        :param src_transform_encoding: (batch_size, seq_len, hidden_dim)
        :param src_encodings: (batch_size, seq_len, ctx_dim)
        :param mask: (batch_size, seq_len)
        :return: att_ctx_vectors: (batch_size, ctx_dim)
        '''
        if self.att_type == "M":
            # (batch_size, att_dim)
            h_t_transform = self.att_hid(h_t).unsqueeze(1)
            att_score = F.tanh(h_t_transform + src_transform_encoding)
            att_weight = self.att_V(att_score).squeeze(2)
        else:
            att_weight = torch.bmm(src_transform_encoding, h_t.unsqueeze(2)).squeeze(2)
        if self.cov_att:
            assert prev_att_sum is not None
            att_weight = att_weight - self.trainable_lambd * prev_att_sum
        if mask is not None:
            inf_mask = Variable(att_weight.data.new(att_weight.size()).fill_(-1e10), requires_grad=False)
            att_weight = att_weight + inf_mask * (1 - mask)

        att_weight = F.softmax(att_weight, dim=1).unsqueeze(1) # (batch_size, 1, seq_len)
        att_ctx = torch.bmm(att_weight, src_encodings).squeeze(1)
        return att_ctx, att_weight.squeeze(1)

    def init_decoder(self, dec_init_cell, new_tensor, batch_size, src_encodings, volatile=False, dec_init_hidden=None):
        if self.init_dec:
            assert dec_init_cell is not None
            init_vec = self.dec_initializer(dec_init_cell)
            if self.args.init_dec == "respect":
                init_vec_h = self.dec_initializer_h(dec_init_hidden)
                h_tm1 = (F.relu(init_vec_h), F.relu(init_vec)) # init_state, init_cell
            else:
                h_tm1 = (F.tanh(init_vec), init_vec)
        else:
            h_tm1 = (self.new_zero_var(new_tensor, batch_size, self.hidden_dim, volatile),
                     self.new_zero_var(new_tensor, batch_size, self.hidden_dim, volatile))

        # (batch_size, src_seq_len, att_dim)
        src_transform_encodings = self.att_src(src_encodings)
        # Initialize the attention vector
        att_tm1 = self.new_zero_var(new_tensor, batch_size, self.input_feed_dim) # TODO: ctx_dim / hidden_dim -> feed ctx / readout
        return h_tm1, src_transform_encodings, att_tm1

    def new_zero_var(self, new_tensor, dim1, dim2, volatile=False):
        return Variable(new_tensor(dim1, dim2).zero_(), requires_grad=False, volatile=volatile)

    def forward(self, *input):
        raise NotImplementedError


class NMT_decoder(decoder_common):
    def __init__(self, args,
                 hidden_dim,
                 emb_dim,
                 ctx_dim,
                 vocab_info,
                 init_dim=-1,
                 att_dim=-1,
                 att_type="D",
                 init_vocab_emb=None,
                 cov_att=False):
        super(NMT_decoder, self).__init__(args, init_dim, hidden_dim, emb_dim, ctx_dim, att_dim, att_type, cov_att=cov_att)

        self.vocab_size = vocab_info["vocab_size"]
        self.bos = vocab_info["bos"]
        self.eos = vocab_info["eos"]
        self.pad = vocab_info["pad"]
        if init_vocab_emb is None:
            self.tgt_emb = nn.Embedding(self.vocab_size, emb_dim, padding_idx=self.pad)
            self.tgt_emb.weight.data.uniform_(-0.1, 0.1)
        else:
            if type(init_vocab_emb) is not nn.Embedding:
                self.tgt_emb = nn.Embedding(vocab_info["vocab_size"], emb_dim, padding_idx=vocab_info["pad"])
                self.tgt_emb.weight.data.copy_(init_vocab_emb)
            else:
                self.tgt_emb = init_vocab_emb
            if args.fix_tgt_emb:
                self.tgt_emb.weight.requires_grad = False

        self.readout = nn.Linear(hidden_dim+ctx_dim, hidden_dim)

        self.logit_reader = nn.Linear(hidden_dim, self.vocab_size, bias=True)
        if args.share_decoder_emb:
            assert hidden_dim == emb_dim
            self.logit_reader.weight = self.tgt_emb.weight

        vocab_mask = torch.ones(self.vocab_size)
        vocab_mask[self.pad] = 0
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False, reduce=True)
        # self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.pad, size_average=False, reduce=True)

    def forward(self, src_encodings, tgt_sents, src_masks=None, tgt_masks=None, dec_init_cell=None, dec_init_hidden=None):
        '''
        :param src_encodings: (batch_size, src_seq_len, ctx_dim)
        :param tgt_sents: (tgt_seq_len, batch_size)
        :param dec_init_vec: (batch_size, init_dim)
        :param src_masks: (batch_size, src_seq_len)
        :return:
        '''
        batch_size = src_encodings.size(0)
        src_seq_len = src_encodings.size(1)
        new_tensor = src_encodings.data.new
        h_tm1, src_transform_encodings, input_feed = self.init_decoder(dec_init_cell, new_tensor, batch_size, src_encodings, dec_init_hidden=dec_init_hidden)

        # (tgt_seq_len, batch_size, emb_dim)
        tgt_word_embs = self.tgt_emb(tgt_sents[:-1 :])
        predictions = []
        prev_sum_att_weights = Variable(new_tensor(batch_size, src_seq_len).zero_())
        for step, y_tm1 in enumerate(tgt_word_embs.split(split_size=1)):
            y_tm1 = y_tm1.squeeze(0)
            input_t = torch.cat([y_tm1, input_feed], 1)
            logit, att_tm1, h_tm1, att_weight, readout = self.step(input_t, h_tm1, src_transform_encodings, src_encodings, prev_sum_att_weights, src_masks)
            if self.args.input_feed == "ctx":
                input_feed = att_tm1
            else:
                input_feed = readout
            prev_sum_att_weights = prev_sum_att_weights + att_weight
            predictions.append(logit)

        # (tgt_seq_len, batch_size, vocab_size) -> (tgt_seq_len * batch_size, vocab_size)
        predictions = torch.stack(predictions).view(-1, self.vocab_size)
        gold_pred = tgt_sents[1:, :].view(-1)
        loss = self.cross_entropy_loss(predictions, gold_pred)
        # loss /= batch_size
        return loss

    def step(self, input_t, h_tm1, src_transform_encodings, src_encodings, prev_sum_att_weights=None, masks=None):
        h_t, c_t = self.dec_rnn(input_t, h_tm1)

        if self.args.drop_h:
            h_t = self.dropout(h_t)

        h_tm1 = (h_t, c_t)
        att_ctx, att_weight = self.attention(h_t, src_transform_encodings, src_encodings, masks, prev_sum_att_weights)

        readout = F.tanh(self.readout(torch.cat([h_t, att_ctx], 1)))
        readout = self.dropout(readout)

        logit = self.logit_reader(readout)

        att_tm1 = att_ctx

        return logit, att_tm1, h_tm1, att_weight, readout

    def beam_search_elem(self, src_encodings, dec_init_cell, max_len=100, dec_init_hidden=None):
        '''
        :param src_encodings: (1, src_seq_len, ctx_dim)
        :param dec_init_vec: (1, init_dim)
        :return:
        '''
        beam_size = self.args.beam_size

        batch_size = 1
        # tensor constructors
        new_float_tensor = src_encodings.data.new
        if self.cuda:
            new_long_tensor = torch.cuda.LongTensor
        else:
            new_long_tensor = torch.LongTensor

        h_tm1, src_transform_encodings, input_feed = self.init_decoder(dec_init_cell, new_float_tensor, batch_size, src_encodings, volatile=True, dec_init_hidden=dec_init_hidden)

        live = 1

        final_scores = []
        final_samples = []

        scores = Variable(new_float_tensor(live).zero_(), volatile=True)
        samples = [[self.bos]]
        prev_sum_att_weights = Variable(new_float_tensor(live, src_encodings.size(1)).zero_(), requires_grad=False)

        end_search = False

        for ii in range(max_len):
            batched_src_encodings = src_encodings.expand(live, src_encodings.size(1), src_encodings.size(2))
            batched_src_transform_encodings = src_transform_encodings.expand(live, src_transform_encodings.size(1), src_transform_encodings.size(2))

            # live
            y_tm1 = Variable(new_long_tensor([s[-1] for s in samples]), volatile=True)
            # (live, emb_dim)
            y_tm1_emb = self.tgt_emb(y_tm1)

            input_t = torch.cat([y_tm1_emb, input_feed], 1)

            logit, att_t, h_t, att_weight_t, readout_t = self.step(input_t, h_tm1, batched_src_transform_encodings, batched_src_encodings, prev_sum_att_weights)
            # (live, vocab_size)
            p_t = F.log_softmax(logit, dim=1)
            # log probability
            flatten_hyp_scores = (scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            topk_scores, topk_inds = torch.topk(flatten_hyp_scores, beam_size * 2)
            cand_indices = topk_inds / self.vocab_size
            cand_words = topk_inds % self.vocab_size

            new_scores = []
            new_hyp_ids = []
            new_samples = []
            select = 0
            for rank_id, (score, cand_id, word_id) in enumerate(zip(topk_scores.cpu().data, cand_indices.cpu().data, cand_words.cpu().data)):
                temp_sample = samples[cand_id] + [word_id]
                if word_id == self.eos:
                    final_samples.append(temp_sample[1:-1])
                    final_scores.append(score)
                    if rank_id == 0:
                        end_search = True
                else:
                    new_scores.append(score)
                    new_hyp_ids.append(cand_id)
                    new_samples.append(temp_sample)
                    select +=1
                    if select == beam_size:
                        break

            live = len(new_hyp_ids)
            samples = new_samples
            scores = Variable(new_float_tensor(new_scores), volatile=True)
            new_hyp_ids = new_long_tensor(new_hyp_ids)

            if end_search and len(final_samples) >= beam_size:
                break
            else:
                end_search = False

            att_tm1 = att_t[new_hyp_ids]
            readout_tm1 = readout_t[new_hyp_ids]

            if self.args.input_feed == "ctx":
                input_feed = att_tm1
            else:
                input_feed = readout_tm1

            prev_sum_att_weights = prev_sum_att_weights[new_hyp_ids]
            att_weight_tm1 = att_weight_t[new_hyp_ids]
            prev_sum_att_weights = prev_sum_att_weights + att_weight_tm1

            h_tm1 = (h_t[0][new_hyp_ids], h_t[1][new_hyp_ids])

        if len(samples) > 0:
            for idx in range(live):
                final_samples.append(samples[idx][1:-1])
                final_scores.append(scores.cpu().data[idx])

        return final_scores, final_samples