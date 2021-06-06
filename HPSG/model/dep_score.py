import torch
import torch.nn as nn

from model.biaffine_attention import BiAAttention
from model.bilinear import BiLinear

class DepScore(nn.Module):
    def __init__(self, num_labels, annotation_dim):
        super(DepScore, self).__init__()

        self.dropout_out = nn.Dropout2d(p=0.33)
        self.arc_h = nn.Linear(annotation_dim, 1024)
        self.arc_c = nn.Linear(annotation_dim, 1024)

        self.attention = BiAAttention()

        self.type_h = nn.Linear(annotation_dim, 250)
        self.type_c = nn.Linear(annotation_dim, 250)
        self.bilinear = BiLinear(250, 250, num_labels)

    def forward(self, outputs, outpute):
        # output from rnn [batch, length, hidden_size]

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        outpute = self.dropout_out(outpute.transpose(1, 0)).transpose(1, 0)
        outputs = self.dropout_out(outputs.transpose(1, 0)).transpose(1, 0)

        # output size [batch, length, arc_space]
        arc_h = nn.functional.relu(self.arc_h(outputs))
        arc_c = nn.functional.relu(self.arc_c(outpute))

        # output size [batch, length, type_space]
        type_h = nn.functional.relu(self.type_h(outputs))
        type_c = nn.functional.relu(self.type_c(outpute))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=0)
        type = torch.cat([type_h, type_c], dim=0)

        arc = self.dropout_out(arc.transpose(1, 0)).transpose(1, 0)
        arc_h, arc_c = arc.chunk(2, 0)

        type = self.dropout_out(type.transpose(1, 0)).transpose(1, 0)
        type_h, type_c = type.chunk(2, 0)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        out_arc = self.attention(arc_h, arc_c)
        out_type = self.bilinear(type_h, type_c)

        return out_arc, out_type