import torch
import torch.nn as nn
import torch.nn.init as init

from model.scaleddotproduct_attention import ScaledDotProductAttention
from model.layer_normalization import LayerNormalization
from model.feature_dropout import FeatureDropout

class LabelAttention(nn.Module):
    """
    Single-head Attention layer for label-specific representations
    """

    def __init__(self, d_model, d_k, d_v, d_l, d_proj, use_resdrop=True, q_as_matrix=False, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        super(LabelAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_l = d_l # Number of Labels
        self.d_model = d_model # Model Dimensionality
        self.d_proj = d_proj # Projection dimension of each label output
        self.use_resdrop = use_resdrop # Using Residual Dropout?
        self.q_as_matrix = q_as_matrix # Using a Matrix of Q to be multiplied with input instead of learned q vectors
        self.combine_as_self = False # Using the Combination Method of Self-Attention

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            if self.q_as_matrix:
                self.w_qs1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
            else:
                self.w_qs1 = nn.Parameter(torch.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
            self.w_ks1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
            self.w_vs1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_v // 2), requires_grad=True)

            if self.q_as_matrix:
                self.w_qs2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
            else:
                self.w_qs2 = nn.Parameter(torch.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
            self.w_ks2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
            self.w_vs2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_v // 2), requires_grad=True)

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            if self.q_as_matrix:
                self.w_qs = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            else:
                self.w_qs = nn.Parameter(torch.FloatTensor(self.d_l, d_k), requires_grad=True)
            self.w_ks = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            self.w_vs = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_v), requires_grad=True)

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        if self.combine_as_self:
            self.layer_norm = LayerNormalization(d_model)
        else:
            self.layer_norm = LayerNormalization(self.d_proj)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            if self.combine_as_self:
                self.proj = nn.Linear(self.d_l * d_v, d_model, bias=False)
            else:
                self.proj = nn.Linear(d_v, d_model, bias=False) # input dimension does not match, should be d_l * d_v
        else:
            if self.combine_as_self:
                self.proj1 = nn.Linear(self.d_l*(d_v//2), self.d_content, bias=False)
                self.proj2 = nn.Linear(self.d_l*(d_v//2), self.d_positional, bias=False)
            else:
                self.proj1 = nn.Linear(d_v//2, self.d_content, bias=False)
                self.proj2 = nn.Linear(d_v//2, self.d_positional, bias=False)
        if not self.combine_as_self:
            self.reduce_proj = nn.Linear(d_model, self.d_proj, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, k_inp=None):
        len_inp = inp.size(0)
        v_inp_repeated = inp.repeat(self.d_l, 1).view(self.d_l, -1, inp.size(-1)) # d_l x len_inp x d_model
        if k_inp is None:
            k_inp_repeated = v_inp_repeated
        else:
            k_inp_repeated = k_inp.repeat(self.d_l, 1).view(self.d_l, -1, k_inp.size(-1)) # d_l x len_inp x d_model

        if not self.partitioned:
            if self.q_as_matrix:
                q_s = torch.bmm(k_inp_repeated, self.w_qs) # d_l x len_inp x d_k
            else:
                q_s = self.w_qs.unsqueeze(1) # d_l x 1 x d_k
            k_s = torch.bmm(k_inp_repeated, self.w_ks) # d_l x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs) # d_l x len_inp x d_v
        else:
            if self.q_as_matrix:
                q_s = torch.cat([
                    torch.bmm(k_inp_repeated[:,:,:self.d_content], self.w_qs1),
                    torch.bmm(k_inp_repeated[:,:,self.d_content:], self.w_qs2),
                    ], -1)
            else:
                q_s = torch.cat([
                    self.w_qs1.unsqueeze(1),
                    self.w_qs2.unsqueeze(1),
                    ], -1)
            k_s = torch.cat([
                torch.bmm(k_inp_repeated[:,:,:self.d_content], self.w_ks1),
                torch.bmm(k_inp_repeated[:,:,self.d_content:], self.w_ks2),
                ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
                ], -1)
        return q_s, k_s, v_s
    
    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        n_head = self.d_l
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        if self.q_as_matrix:
            q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        else:
            q_padded = q_s.repeat(mb_size, 1, 1) # (d_l * mb_size) x 1 x d_k
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=torch.bool)

        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            if self.q_as_matrix:
                q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
            k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
            v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
            invalid_mask[i, :end-start].fill_(False)

        if self.q_as_matrix:
            q_padded = q_padded.view(-1, len_padded, d_k)
            attn_mask = invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1)
        else:
            attn_mask = invalid_mask.unsqueeze(1).repeat(n_head, 1, 1)
        
        output_mask = (~invalid_mask).repeat(n_head, 1)

        return(
            q_padded,
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            attn_mask,
            output_mask,
            )

    def combine_v(self, outputs):
        # Combine attention information from the different labels
        d_l = self.d_l
        outputs = outputs.view(d_l, -1, self.d_v) # d_l x len_inp x d_v

        if not self.partitioned:
            # Switch from d_l x len_inp x d_v to len_inp x d_l x d_v
            if self.combine_as_self:
                outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, d_l * self.d_v)
            else:
                outputs = torch.transpose(outputs, 0, 1)#.contiguous() #.view(-1, d_l * self.d_v)
            # Project back to residual size
            outputs = self.proj(outputs) # Becomes len_inp x d_l x d_model
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:,:,:d_v1]
            outputs2 = outputs[:,:,d_v1:]
            if self.combine_as_self:
                outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, d_l * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, d_l * d_v1)
            else:
                outputs1 = torch.transpose(outputs1, 0, 1)#.contiguous() #.view(-1, d_l * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1)#.contiguous() #.view(-1, d_l * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
                ], -1)#.contiguous()

        return outputs

    def forward(self, inp, batch_idxs, k_inp=None):
        residual = inp # len_inp x d_model
        len_inp = inp.size(0)

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, k_inp=k_inp)
        # d_l x len_inp x d_k
        # q_s is d_l x 1 x d_k

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)
        # q_padded, k_padded, v_padded: (d_l * batch_size) x max_len x d_kv
        # q_s is (d_l * batch_size) x 1 x d_kv

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
            )
        # outputs_padded: (d_l * batch_size) x max_len x d_kv
        # in LAL: (d_l * batch_size) x 1 x d_kv
        # on the best model, this is one value vector per label that is repeated max_len times
        if not self.q_as_matrix:
            outputs_padded = outputs_padded.repeat(1,output_mask.size(-1),1)
        outputs = outputs_padded[output_mask]
        # outputs: (d_l * len_inp) x d_kv or LAL: (d_l * len_inp) x d_kv
        # output_mask: (d_l * batch_size) x max_len
        torch.cuda.empty_cache()
        outputs = self.combine_v(outputs)
        # outputs: len_inp x d_l x d_model, whereas a normal self-attention layer gets len_inp x d_model
        if self.use_resdrop:
            if self.combine_as_self:
                outputs = self.residual_dropout(outputs, batch_idxs)
            else:
                outputs = torch.cat([self.residual_dropout(outputs[:,i,:], batch_idxs).unsqueeze(1) for i in range(self.d_l)], 1)
        if self.combine_as_self:
            outputs = self.layer_norm(outputs + inp)
        else:
            for l in range(self.d_l):
                outputs[:, l, :] = outputs[:, l, :] + inp
            
            outputs = self.reduce_proj(outputs) # len_inp x d_l x d_proj
            outputs = self.layer_norm(outputs) # len_inp x d_l x d_proj
            outputs = outputs.view(len_inp, -1).contiguous() # len_inp x (d_l * d_proj)
        
        return outputs, attns_padded