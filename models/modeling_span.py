# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCELoss
import torch.nn.functional as F
from utils.loss_utils import ReverseLayerF

class Boundary_Alignment(BertPreTrainedModel):
    def __init__(self, config, span_num_labels, device):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.span_num_labels = span_num_labels
        # self.type_num_labels_src = type_num_labels_src+1
        # self.type_num_labels_tgt = type_num_labels_tgt+1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        # self.span = nn.Parameter(torch.randn(type_num_labels, config.hidden_size, span_num_labels))
        # self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)
        self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)
        # self.classifier_m_src = nn.Linear(config.hidden_size, type_num_labels_src)
        # self.classifier_m_tgt = nn.Linear(config.hidden_size, type_num_labels_tgt)
        # self.discriminator = nn.Linear(config.hidden_size, 2)

        self.ent_type_size = 1
        self.inner_dim = 64
        self.RoPE = True

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2)  # (inner_dim * 2, ent_type_size * 2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        segment_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_ids=None,
        label_mask=None,
        order=None,
        trainf=True
    ): # inputs_fw = {"input_ids": batch[0], "attention_mask": batch[2], "segment_ids": batch[4],\
                            # "label_ids": batch[6], "label_mask": batch[8], "order": batch[10]}
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=segment_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        # print(outputs[0][0])
        # # print(outputs[1].size())
        # # print(len(outputs[2]))
        # print(outputs[2][-1][0])
        # print(outputs[2][0].size())
        # print(outputs[2][1].size())
        # exit()
        final_embedding = outputs[0] # B, L, D
        #####
        boundary_v = self.boundary_matrix(final_embedding, attention_mask) # return n (non masked token pairs)

        #####
        sequence_output = self.dropout(final_embedding)
        h_dim = sequence_output.size()[-1]
        logits_bio = self.classifier_bio(sequence_output) # B, L, C
        if label_ids is not None:
            active_loss = (label_ids.view(-1)>=0) & (attention_mask.view(-1) == 1)
            # active_seq_rep = sequence_output.view(-1, h_dim)[active_loss] 

            # implicit alignment
            # logits_bio = self.classifier_bio(active_seq_rep) # B*L, C
            loss_fct = CrossEntropyLoss()
            active_logits_bio = logits_bio.view(-1, self.span_num_labels)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss_bio = loss_fct(active_logits_bio, active_labels)
        else:
            active_loss = attention_mask.view(-1) == 1
            logits_bio = logits_bio.view(-1, self.span_num_labels)[active_loss]
            loss_bio = None

        return (loss_bio, sequence_output, logits_bio, boundary_v)
        
        # outputs = (logits_bio, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        # # outputs = (loss_domain, logits_bio, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here

        # if label_ids is not None:
        #     # logits = self.logsoftmax(logits)
        #     # Only keep active parts of the loss
        #     active_loss = True
        #     if attention_mask is not None:
        #         active_loss = attention_mask.view(-1) == 1
            
        #     active_logits = logits_bio.view(-1, self.span_num_labels)[active_loss]

        #     loss_fct = CrossEntropyLoss()
        #     if attention_mask is not None:
        #         active_labels = label_ids.view(-1)[active_loss]
        #         loss_bio = loss_fct(active_logits, active_labels)
        #     else:
        #         loss_bio = loss_fct(logits_bio.view(-1, self.span_num_labels), label_ids.view(-1))

        #     outputs = (loss_bio, active_logits,) + outputs

        # return outputs
        
        # # sequence_output2 = self.dropout(final_embedding)
        # # reverse_feature = ReverseLayerF.apply(sequence_output2, alpha)
        # # logits_domain = self.discriminator(reverse_feature) # B, L, 2
        # # loss_fct = CrossEntropyLoss()
        # # logits_size = logits_domain.size()
        # # labels_domain = torch.zeros(logits_size[0]*logits_size[1]).long().to(self.device_)
        # # if tgt:
        # #     logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
        # #     # labels_domain = labels_domain + 1
        # #     # loss_domain = loss_fct(logits_domain.view(-1, 2), labels_domain)

        # # else:
        # #     logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
        # #     # loss_domain = loss_fct(logits_domain.view(-1, 2), labels_domain)
        # logits_bio = self.classifier_bio(sequence_output1) # B, L, C
        
        # outputs = (logits_bio, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        # # outputs = (loss_domain, logits_bio, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here

        # if labels_bio is not None:
        #     # logits = self.logsoftmax(logits)
        #     # Only keep active parts of the loss
        #     active_loss = True
        #     if attention_mask is not None:
        #         active_loss = attention_mask.view(-1) == 1
            
        #     active_logits = logits_bio.view(-1, self.span_num_labels)[active_loss]

        #     loss_fct = CrossEntropyLoss()
        #     if attention_mask is not None:
        #         active_labels = labels_bio.view(-1)[active_loss]
        #         loss_bio = loss_fct(active_logits, active_labels)
        #     else:
        #         loss_bio = loss_fct(logits_bio.view(-1, self.span_num_labels), labels_bio.view(-1))

        #     outputs = (loss_bio, active_logits,) + outputs

        # return outputs
    
    def contrastive_loss(self, h_fw, h_bw, order_fw, order_bw):
        # h_fw: N, L_f, d
        # h_bw: N, L_b, d
        # order_fw/order_bw: N, L
        N_f, L_f = order_fw.size()
        N_b, L_b = order_bw.size()
        assert N_f==N_b
        order_fw_expd = order_fw.unsqueeze(2).expand(N_f, L_f, L_b) # N_f, L_f, L_b
        order_bw_expd = order_bw.unsqueeze(1).expand(N_f, L_f, L_b)
        order_mask = (order_fw_expd==order_bw_expd)&(order_fw_expd>0)&(order_bw_expd>0) # N_f, L_f, L_b
        s = torch.bmm(h_fw, h_bw.permute(0,2,1)) # N, L_f, L_b
        # h_fw_expd = h_fw.unsqueeze(2).expand()
        f = torch.norm(h_fw, None, 2).unsqueeze(2).expand(N_f, L_f, L_b) # N, L_f, L_b
        b = torch.norm(h_bw, None, 2).unsqueeze(1).expand(N_f, L_f, L_b) # N, L_f, L_b
        s = s/(f*b) # N, L_f, L_b
        s_norm = F.log_softmax(s/0.1, dim=-1) # N, L_f, L_b
        loss_cs = torch.mean(-s_norm.view(-1)[order_mask.view(-1)])
        # loss_funct = NLLLoss()
        # loss_funct()
        # output = F.cosine_similarity(input1, input2, dim=0)
        return loss_cs

    def loss(self, loss_bio, logits_type, delta=0.1):
        # loss_bio: B*L
        # logits_type: B*L, C
        logits_type = torch.softmax(logits_type.detach()/delta, dim=-1)
        weight = 1.5-logits_type[:, -1]
        loss = torch.mean(loss_bio*weight)
        return loss

    def adv_attack(self, emb, loss, epsilon):
        loss_grad = torch.autograd.grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, dim=2))
        perturbed_sentence = emb + epsilon * (loss_grad/(loss_grad_norm.unsqueeze(2)+1e-5))
        
        return perturbed_sentence
    
    def boundary_matrix(self, last_hidden_state, attention_mask):
        # last_hidden_state: [bz, seq_len, hidden_dim]
        # print(last_hidden_state.size())
        outputs = self.dense_1(last_hidden_state) # [bz, seq_len, 2*inner_dim]
        # print(outputs.size())
        qw, kw = outputs[..., ::2], outputs[..., 1::2]
        # print(qw.size(), kw.size())
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1) # e.g. [0.34, 0.90] -> [0.34, 0.34, 0.90, 0.90]
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        
        # return qw, kw
        # print(qw.size(), kw.size())
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        # print(logits.size())
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        # print(bias.size())
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 
        # batch_size, 1, seq_len, seq_len
        # print(logits.size())
        # exit()
        padding_mask = torch.bmm(attention_mask.unsqueeze(2).float(),attention_mask.unsqueeze(1).float()) # batch_size, seq_len, seq_len
        logits = logits.squeeze(1)[padding_mask==1]

        return logits

class SinusoidalPositionEmbedding(nn.Module):

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)

# class Span_Learner(BertPreTrainedModel):
#     def __init__(self, config, span_num_labels, type_num_labels, device):
#         super().__init__(config)

#         self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.span_num_labels = span_num_labels
#         self.type_num_labels = type_num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.hidden_size = config.hidden_size
#         self.span = nn.Parameter(torch.randn(type_num_labels, config.hidden_size, span_num_labels))
#         # self.classifier_meta = nn.Linear(config.hidden_size, num_labels)

#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         label_mask=None,
#     ):
#         # print(input_ids)
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
#         final_embedding = outputs[0] # B, L, D
#         # print(final_embedding.size())
#         sequence_output = self.dropout(final_embedding)
#         seq_embed = sequence_output.view(-1, self.hidden_size) # B*L, D
#         seq_size = seq_embed.size()
#         logits = torch.bmm(seq_embed.unsqueeze(0).expand(self.type_num_labels, 
#                         seq_size[0], seq_size[1]), self.span).permute(1, 0, 2) # B*L, type_num, span_num
#         # logits = self.classifier_meta(sequence_output) # B, L, C
#         # print(logits.size())

#         return logits
