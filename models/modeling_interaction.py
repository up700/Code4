# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCELoss
from utils.loss_utils import ReverseLayerF
import torch.nn.functional as F
import math

class Span_Detector(BertPreTrainedModel):
    def __init__(self, config, span_num_labels, device):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.span_num_labels = span_num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        self.classifier_map = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)
        self.label_enhanc = Label_Attention(self.classifier_bio)
        self.biaf_att = Biaffine(config.hidden_size, config.hidden_size, 1)
        self.kl_loss = KLDivLoss(reduction="batchmean")

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_bio=None,
        tgt=True
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )

        final_embedding = outputs[0] # B, L, D
        sequence_output1 = self.dropout(final_embedding)

        enhanced_rep = self.label_enhanc(sequence_output1)

        return enhanced_rep
    
    def interaction(self, label_enhanced_span, label_enhanced_type, kl_fw, kl_bw, attention_mask=None):
        """
        label_enhanced_span: B, L1, d1
        label_enhanced_type: B, L2, d2
        
        """
        att_score = self.biaf_att(label_enhanced_span, label_enhanced_type).squeeze_(3)
        interaction_rep = torch.bmm(att_score, label_enhanced_type) # B, L1, d2
        rep_combined = torch.cat((label_enhanced_span, interaction_rep), dim=-1) # B, L1, d1+d2
        reduced_rep = self.classifier_map(rep_combined)
        reduced_rep = self.dropout(reduced_rep)
        logits_bio = self.classifier_bio(reduced_rep)

        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            logits_bio = logits_bio.view(-1, self.span_num_labels)[active_loss]

            kl_pred = F.log_softmax(logits_bio, dim=-1)

            kl_loss_fw = self.kl_loss(kl_pred, kl_fw)
            kl_loss_bw = self.kl_loss(kl_pred, kl_bw)
        else:
            kl_loss_fw = None
            kl_loss_bw = None

        return kl_loss_fw, kl_loss_bw, logits_bio


class Type_Predictor(BertPreTrainedModel):
    def __init__(self, config, type_num_labels, device):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.type_num_labels = type_num_labels+1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        self.classifier_map = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.classifier_type = nn.Linear(config.hidden_size, type_num_labels+1)
        self.label_enhanc = Label_Attention(self.classifier_type)
        self.biaf_att = Biaffine(config.hidden_size, config.hidden_size, 1)
        self.kl_loss = KLDivLoss(reduction="batchmean")

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_type=None,
        logits_bio=None,
        tgt=True
        # reduction="none"
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        final_embedding = outputs[0] # B, L, D
        sequence_output2 = self.dropout(final_embedding)
        enhanced_rep = self.label_enhanc(sequence_output2)

        return enhanced_rep
    
    def interaction(self, label_enhanced_type, label_enhanced_span, kl_fw, kl_bw, attention_mask=None):
        """
        label_enhanced_span: B, L1, d1
        label_enhanced_type: B, L2, d2
        
        """
        att_score = self.biaf_att(label_enhanced_type, label_enhanced_span).squeeze_(3)
        interaction_rep = torch.bmm(att_score, label_enhanced_span) # B, L1, d2
        rep_combined = torch.cat((label_enhanced_type, interaction_rep), dim=-1) # B, L1, d1+d2
        reduced_rep = self.classifier_map(rep_combined)
        reduced_rep = self.dropout(reduced_rep)
        logits_tp = self.classifier_type(reduced_rep)

        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            logits_tp = logits_tp.view(-1, self.type_num_labels)[active_loss]

            kl_pred = F.log_softmax(logits_tp, dim=-1)

            kl_loss_fw = self.kl_loss(kl_pred, kl_fw)
            kl_loss_bw = self.kl_loss(kl_pred, kl_bw)
        else:
            kl_loss_fw = None
            kl_loss_bw = None

        return kl_loss_fw, kl_loss_bw, logits_tp


class Label_Attention(nn.Module):
    def __init__(self, emb):
        super(Label_Attention, self).__init__()

        self.W_emb = emb.weight

    def forward(self, input_):
        _score = torch.matmul(input_, self.W_emb.t())
        _probs = nn.Softmax(dim=-1)(_score)
        _res = torch.matmul(_probs, self.W_emb)

        return _res

class Biaffine(nn.Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int):
        super().__init__()
        self.bilinear = PairwiseBilinear(in1_features + 1, in2_features + 1, out_features)
        self.bilinear.weight.data.zero_()
        self.bilinear.bias.data.zero_()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], dim=input1.dim() - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], dim=input2.dim() - 1)
        return self.bilinear(input1, input2)


class PairwiseBilinear(nn.Module):

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, out_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        # (b * n1, d1) @ (d1, out * d2) => (b * n1, out * d2)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        # (b, n1 * out, d2) @ (b, d2, n2) => (b, n1 * out, n2)
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y  # (b, n1, n2, out)

    def extra_repr(self) -> str:
        return "in1_features={}, in2_features={}, out_features={}, bias={}".format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )
        
