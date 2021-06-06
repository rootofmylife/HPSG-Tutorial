import torch
import torch.nn as nn

class BiAAttention(nn.Module):
    '''
    Bi-Affine attention layer.
    '''

    def __init__(self, hparams):
        super(BiAAttention, self).__init__()
        self.hparams = hparams

        self.dep_weight = nn.Parameter(torch.FloatTensor(hparams.d_biaffine + 1, hparams.d_biaffine + 1))
        nn.init.xavier_uniform_(self.dep_weight)

    def forward(self, input_d, input_e, input_s = None):

        score = torch.matmul(torch.cat(
            [input_d, torch.FloatTensor(input_d.size(0), 1).fill_(1).requires_grad_(False)],
            dim=1), self.dep_weight)
        score1 = torch.matmul(score, torch.transpose(torch.cat(
            [input_e, torch.FloatTensor(input_e.size(0), 1).fill_(1).requires_grad_(False)],
            dim=1), 0, 1))

        return score1