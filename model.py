from utils import *


def attention(h_t, src_transform_encoding, src_encodings, mask=None):
    '''
    :param h_t: (batch_size, hidden_dim)
    :param src_transform_encoding: (batch_size, seq_len, hidden_dim)
    :param src_encodings: (batch_size, seq_len, ctx_dim)
    :param mask: (batch_size, seq_len)
    :return: att_ctx_vectors: (batch_size, ctx_dim)
    '''

    att_weight = torch.bmm(src_transform_encoding, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        inf_mask = Variable(att_weight.data.new(att_weight.size()).fill_(-1e10), requires_grad=False)
        att_weight = att_weight + inf_mask * (1 - mask)

    att_weight = F.softmax(att_weight, dim=1).unsqueeze(1)  # (batch_size, 1, seq_len)
    att_ctx = torch.bmm(att_weight, src_encodings).squeeze(1)
    return att_ctx, att_weight.squeeze(1)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.save_to = args.save_to
        self.load_from = args.load_from
        self.use_cuda = args.cuda

    def save(self):
        print('Save model params to %s' % self.save_to)
        torch.save(self.state_dict(), self.save_to)

    def load(self):
        print('Load model params from %s' % self.load_from)
        self.load_state_dict(torch.load(self.load_from))